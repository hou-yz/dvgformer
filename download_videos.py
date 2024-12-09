import argparse
from distutils.util import strtobool
import multiprocessing
import tqdm
import re
import os
import sys
import datetime
import logging
import shutil
import time
import h5py
import pandas as pd
import numpy as np
from src.preparation.youtube_download import download_youtube_video
from src.preparation.split_scene_frame import _recons_images_from_ffmpeg


def setup_logging(log_fname='app.log', logger_name='root', level=logging.INFO):
    os.makedirs(os.path.dirname(log_fname), exist_ok=True)
    logger = logging.getLogger(logger_name)
    # Set to the lowest level to capture all messages
    logger.setLevel(level)
    # disable propagation to avoid duplicate logs from the root logger
    logger.propagate = False

    # Check if handlers are already configured
    if not logger.handlers:
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(message)s')

        # Setup file handler
        fh = logging.FileHandler(log_fname)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Setup console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def download_video_process(log_fname, download_path, remove_existing_download, video_resolution, video_ids, download_queue, extract_queue, error_queue):
    logger = setup_logging(log_fname, f"{'download':^10}")
    while True:
        # time.sleep(5)
        video_id = download_queue.get()
        if video_id is None:
            logger.info("No more items to process, exiting.")
            break
        try:
            # Code to download a video
            t0 = time.time()
            need_download = True
            if os.path.exists(f'{download_path}/{video_id}/video.mp4'):
                if remove_existing_download:
                    os.remove(f'{download_path}/{video_id}/video.mp4')
                    os.remove(f'{download_path}/{video_id}/data.json')
                else:
                    need_download = False
            if need_download:
                download_youtube_video(download_path,
                                       video_id,
                                       video_resolution,
                                       quiet=True)
                t1 = time.time()
                logger.info(
                    f'{video_id:<30}{video_ids.index(video_id)+1}/{len(video_ids)}\t\t\t{t1 - t0:.2f}s')
            else:
                logger.info(
                    f'{video_id:<30}{video_ids.index(video_id)+1}/{len(video_ids)}\talready downloaded')
            # update the extract queue
            extract_queue.put(f'{download_path}/{video_id}')
        except Exception as e:
            msg = f'{video_id:<30}{video_ids.index(video_id)+1}/{len(video_ids)}\terror! {str(e)}'
            logger.info(msg)
            error_queue.put(msg)
        # mark the download task as done
        download_queue.task_done()


def queue_update_process(log_fname, hdf5_fpath, extract_queue, post_queue, error_queue):
    logger = setup_logging(log_fname, f"{'extract':^10}")
    while True:
        # time.sleep(5)
        video_folder = extract_queue.get()
        if video_folder is None:
            logger.info("No more items to process, exiting.")
            break
        # Code to extract a video
        video_id = os.path.basename(video_folder)
        download_path = os.path.dirname(hdf5_fpath)
        try:
            # scene clip info
            scene_df = pd.read_csv(f'{video_folder}/scenes_updated.csv',
                                   index_col=0)
            scene_df = pd.concat(
                [pd.to_timedelta(scene_df['start']).dt.total_seconds(),
                 pd.to_timedelta(scene_df['end']).dt.total_seconds()],
                axis=1)
            for fname in os.listdir(video_folder):
                scene = fname.split('-recons')[0].replace('scene', '')
                if '-score' in fname and fname.endswith('.csv'):
                    start_s, end_s = scene_df['start'][scene], scene_df['end'][scene]
                    # Read data from HDF5 file
                    recons_df = pd.read_csv(
                        f'{video_folder}/{fname}', comment='#')
                    recons_array = recons_df.to_numpy()
                    post_queue.put(
                        (f'{download_path}/{video_id}/{fname}',
                            recons_array[:, :1], (start_s, end_s)))
        except Exception as e:
            msg = f'{video_id:<30}error! {str(e)}'
            logger.info(msg)
            error_queue.put(msg)
        # mark the extraction task as done
        extract_queue.task_done()


def extraction_process(log_fname, original_fps, dataset_fps, frame_resolution, fix_image_width, video_ids, post_queue, archive_queue, error_queue):
    logger = setup_logging(log_fname, f"{'post':^10}")
    while True:
        post = post_queue.get()
        if post is None:
            logger.info("No more items to process, exiting.")
            break
        else:
            result_fpath, recons_array, (start_s, end_s) = post
        video_folder = os.path.dirname(result_fpath)
        video_id, fname = result_fpath.split(os.path.sep)[-2:]
        scene_dir = fname.split('-recons')[0]
        clip_name = os.path.sep.join([video_id, scene_dir])
        try:
            t0 = time.time()
            i = int(re.findall(r'recons(\d+)', result_fpath)[0])
            score = int(re.findall(r'score(\d+)', result_fpath)[0])
            valid = '_invalid' not in result_fpath

            # re-generate frames if requested
            if frame_resolution > 0 and score and valid:
                frames_folder = f'{video_folder}/{scene_dir}-recons{i}-frames/'
                if os.path.exists(frames_folder):
                    shutil.rmtree(frames_folder)
                os.makedirs(frames_folder)
                _recons_images_from_ffmpeg(
                    video_folder, start_s, end_s, recons_array,
                    original_fps, dataset_fps, frame_resolution,
                    fix_image_width=fix_image_width,
                    save_folder=frames_folder)
                # add extracted frames to combine into hdf5
                archive_queue.put(frames_folder)

            t3 = time.time()
            logger.info(
                f'{clip_name:<30}{video_ids.index(video_id)+1}/{len(video_ids)}\t\t\t{t3 - t0:.2f}s')
        except Exception as e:
            msg = f'{clip_name:<30}error! \t{e}'
            logger.info(msg)
            error_queue.put(msg)
        # mark the reconstruction task as done
        post_queue.task_done()


def writeh5_process(log_fname, write_fname, download_path, archive_queue):
    logger = setup_logging(log_fname, f"{'writeHDF5':^10}")
    with h5py.File(write_fname, 'a') as hdf:
        while True:
            tmpfile_path = archive_queue.get()
            if tmpfile_path is None:
                logger.info("No more items to process, exiting.")
                break
            # Check if the path is a directory
            if os.path.isdir(tmpfile_path):
                # Add the directory and all its subcontents
                for root, dirs, files in os.walk(tmpfile_path):
                    for file in files:
                        full_path = os.path.join(root, file)
                        arcname = os.path.relpath(
                            full_path, start=download_path)
                        with open(full_path, 'rb') as file:
                            file_data = file.read()
                        # Store data as a byte array (uint8)
                        if arcname in hdf:
                            del hdf[arcname]
                        hdf.create_dataset(arcname, data=np.frombuffer(
                            file_data, dtype='uint8'))
                        os.remove(full_path)
                    logger.debug(f"Added directory {tmpfile_path} to hdf5.")
                # remove the directory
                os.rmdir(tmpfile_path)
            else:
                # Just add the file
                with open(tmpfile_path, 'rb') as file:
                    file_data = file.read()
                # Store data as a byte array (uint8)
                arcname = os.path.relpath(tmpfile_path, start=download_path)
                if arcname in hdf:
                    del hdf[arcname]
                hdf.create_dataset(arcname, data=np.frombuffer(
                    file_data, dtype='uint8'))
                os.remove(tmpfile_path)
                logger.debug(f"Added file {tmpfile_path} to hdf5.")


def main(args):
    t0 = time.time()

    hdf5_fpath = os.path.expanduser(args.hdf5_fpath)
    download_path = os.path.dirname(hdf5_fpath)
    # Set up logging
    current_time_str = f'{datetime.datetime.today():%Y-%m-%d_%H-%M-%S}'
    log_fname = f'{download_path}/{current_time_str}.log'
    logger = setup_logging(log_fname, f"{'root':^10}")
    logger.info(vars(args))

    # Create a Manager and use it to create a Queue
    manager = multiprocessing.Manager()
    # create a dict that is shared between processes
    download_queue = manager.Queue()
    extract_queue = manager.Queue()
    post_queue = manager.Queue(maxsize=16)
    archive_queue = manager.Queue(maxsize=16)
    error_queue = manager.Queue()
    # workers
    download_workers = []
    extract_workers = []

    video_ids = []
    with h5py.File(hdf5_fpath) as hdf:
        for video_id in tqdm.tqdm(hdf.keys()):
            if 'scenes_updated.csv' in hdf[video_id]:
                recons_fnames = [fname for fname in sorted(
                    hdf[video_id].keys()) if '-score' in fname and fname.endswith('.csv')]
                if len(recons_fnames) == 0:
                    continue
                os.makedirs(f'{download_path}/{video_id}', exist_ok=True)
                recons_fnames.append('scenes_updated.csv')
                for fname in recons_fnames:
                    if os.path.exists(f'{download_path}/{video_id}/{fname}'):
                        os.remove(f'{download_path}/{video_id}/{fname}')
                    with open(f'{download_path}/{video_id}/{fname}', 'wb') as f:
                        f.write(hdf[video_id][fname][:])
                video_ids.append(video_id)

    if args.download:
        # queue for download
        for video_id in video_ids:
            download_queue.put(video_id)
        # Signal download workers to stop
        for _ in range(args.num_download_workers):
            download_queue.put(None)
        # Start download workers
        for _ in range(args.num_download_workers):
            worker = multiprocessing.Process(target=download_video_process,
                                             args=(log_fname, download_path, args.remove_existing_download, args.video_resolution, video_ids, download_queue, extract_queue, error_queue))
            worker.start()
            download_workers.append(worker)

    if args.extract:
        # queuing helper
        queue_worker = multiprocessing.Process(target=queue_update_process,
                                               args=(log_fname, hdf5_fpath, extract_queue, post_queue, error_queue))
        queue_worker.start()
        if not args.download:
            for video_id in video_ids:
                extract_queue.put(f'{download_path}/{video_id}')

        # extraction workers
        for _ in range(args.num_extract_workers):
            worker = multiprocessing.Process(target=extraction_process,
                                             args=(log_fname, args.colmap_fps, args.dataset_fps, args.frame_resolution, args.fixed_width, video_ids, post_queue, archive_queue, error_queue))
            worker.start()
            extract_workers.append(worker)

        # archive worker
        archive_worker = multiprocessing.Process(target=writeh5_process,
                                                 args=(log_fname, hdf5_fpath, download_path, archive_queue))
        archive_worker.start()

    # Wait for download workers to finish
    if args.download:
        for worker in download_workers:
            worker.join()

        hours, remainder = divmod(time.time() - t0, 3600)
        minutes, seconds = divmod(remainder, 60)
        logger.info(
            f'step 1 finished! {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds')

    # Signal & wait extraction process workers to stop
    if args.post:
        extract_queue.put(None)
        queue_worker.join()

        for _ in extract_workers:
            post_queue.put(None)
        for worker in extract_workers:
            worker.join()

        archive_queue.put(None)
        archive_worker.join()

        hours, remainder = divmod(time.time() - t0, 3600)
        minutes, seconds = divmod(remainder, 60)
        logger.info(
            f'step 2 finished! {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds')

    # After all workers are done, check for errors
    while not error_queue.empty():
        error = error_queue.get()
        logger.info(f"Error in worker process: {error}")


if __name__ == "__main__":
    # common settings
    parser = argparse.ArgumentParser(
        description='parallel creation of youtube drone video dataset with 3D reconstruction')
    parser.add_argument('--hdf5_fpath', type=str, required=True)
    # num workers
    parser.add_argument('--num_download_workers', type=int, default=1)
    parser.add_argument('--num_extract_workers', type=int, default=4)
    # download
    parser.add_argument('--download',
                        type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument('--remove_existing_download',
                        type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('--video_resolution', type=int,
                        default=1920, help='default: 1080p=1920x1080')
    parser.add_argument('--colmap_fps', type=int, default=15)
    # extract
    parser.add_argument('--extract',
                        type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument('--frame_resolution',
                        type=int, default=480,
                        help=('whether to remove high-resolution frames for colmap and regenerate low resolution frames.'
                              '0: only remove existing frames;'
                              '> 0: regenerate frames at specified resolution'))
    parser.add_argument('--fixed_width',
                        type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('--dataset_fps', type=int, default=3)

    args = parser.parse_args()
    main(args)
