# Training Datasets

We provide the Colmap 3D reconstruction results and the filtered camera movement sequences in our DroneMotion-99k dataset. You can download either a minimal dataset with 10 videos and 129 sequences [link](https://1drv.ms/u/c/dfb1b9d32643ecdc/ERIEM1bBgvVHtqgyN4T-7qoBmiHYaHcAdUUz5McREVuI_w?e=qwOBge) or the full dataset with 13,653 videos and 99,003 camera trajectories [link](https://1drv.ms/u/c/dfb1b9d32643ecdc/EcHhl1KtZrdHn4wkDJ9Kcg4BtwQCP3f3hKUHS7PArhprnw?e=SRkFjl). 

After downloading the training data, your folder should look like this
```
dvgformer/
├── youtube_drone_videos/
│   ├── dataset_full.h5/
│   └── dataset_mini.h5/
├── src/
├── README.md
...
```

Due to the YouTube policy, we cannot share the video MP4s or the frames. As an alternative, we include a python script `download_videos.py` that can help you automatically download the videos and extract the frames. 
```python
python download_videos.py --hdf5_fpath youtube_drone_videos/dataset_mini.h5
python download_videos.py --hdf5_fpath youtube_drone_videos/dataset_full.h5
```
This should update your downloaded HDF5 dataset file with the video frames. 

You can also adjust the number of workers for the download process or the frame extraction process in `download_videos.py` by specifying `--num_download_workers` or `--num_extract_workers`.
