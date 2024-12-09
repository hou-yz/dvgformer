import io
import os
import re
import glob
import tarfile
import h5py
import numpy as np


def check_tar_gz_integrity(file_path):
    try:
        with tarfile.open(file_path, 'r:gz') as tar:
            tar.getmembers()  # List all members in the archive
        return True
    except tarfile.TarError:
        return False


class ManagedBytesIO(io.BytesIO):
    def __init__(self):
        super().__init__()
        self._is_closed = False

    def close(self):
        # Instead of closing the file, we set a flag
        self._is_closed = True

    def reopen(self):
        self._is_closed = False

    def real_close(self):
        # Actually close the BytesIO object
        super().close()

    @property
    def closed(self):
        """Return the state of the 'closed' flag rather than the actual stream state."""
        return self._is_closed

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()  # Call our custom close, which does not actually close the buffer


class FlexibleFileSystem:
    '''
    A class to handle file operations on both real filesystem, tar files, and h5 files.
    '''

    def __init__(self, fpath):
        # find either directory or tar/h5 file starting with `archive_path`
        if os.path.isdir(fpath):
            archive_candidates = []
        else:
            # if there is no dir under fpath, then search for tar/h5 files with the same initial
            # if no tar/h5 files found either, then raise an error
            archive_candidates = glob.glob(f'{fpath}*')
            if not archive_candidates:
                raise FileNotFoundError(
                    f'No such file or directory: {fpath}')
        # find the first tar/h5 file
        self.archive_path = None
        for candidate in archive_candidates:
            if candidate.endswith(('.tar', '.tar.gz', 'h5')):
                self.archive_path = candidate
                break
        if self.archive_path is not None:
            match = re.search(r'(\.tar\.gz|\.tar|\.h5)$', self.archive_path)
            self.archive_extension = match.group(1)
            self.dir_path = os.path.dirname(self.archive_path)
        else:
            self.archive_extension = None
            self.dir_path = None
        self.modified = False
        self.tar_file = None
        self.h5_file = None
        if self.archive_extension is not None:
            if 'tar' in self.archive_extension:
                self.tar_file = tarfile.open(self.archive_path, 'r')
            elif 'h5' in self.archive_extension:
                # Open the HDF5 file in SWMR mode
                self.h5_file = h5py.File(self.archive_path, 'r', swmr=True)
        self.memory_files = {}  # To store modified or new files in memory

    def _rel_path(self, path):
        # Compute path relative to the root of the tar or directory
        if self.archive_extension:
            return os.path.relpath(path, start=self.dir_path).replace(self.archive_extension, '')
        else:
            return path

    def exists(self, path):
        if self.tar_file:
            path = self._rel_path(path)
            if path in self.memory_files:
                return True
            # Check if the exact path exists (for files or exact directory names)
            try:
                self.tar_file.getmember(path)
                return True
            except KeyError:
                pass
            # Additionally check if it's a directory by looking for members starting with this path
            path = path.strip('/') + '/'  # Ensure directory-like path
            for member in self.tar_file.getmembers():
                if member.name.startswith(path):
                    return True
            return False
        elif self.h5_file:
            path = self._rel_path(path)
            return path in self.h5_file
        else:
            return os.path.exists(path)

    def listdir(self, path):
        if self.tar_file:
            path = self._rel_path(path)
            if not path.endswith('/'):
                path += '/'
            if path == './':
                path = ''
            entries = set()
            # for member in self.tar_file.getmembers():
            #     if member.name.startswith(path):
            #         relative_path = member.name[len(path):]
            fpaths = ([member.name for member in self.tar_file.getmembers()] +
                      list(self.memory_files.keys()))
            for fpath in fpaths:
                if fpath.startswith(path):
                    relative_path = fpath[len(path):]
                    # Split and get the first component to avoid listing subdirectories
                    entry = relative_path.split('/', 1)[0]
                    if entry:
                        entries.add(entry)
            return list(entries)
        elif self.h5_file:
            path = self._rel_path(path)
            if path in self.h5_file:
                return list(self.h5_file[path])
            else:
                return []
        else:
            return os.listdir(path)

    def open(self, path, mode='r'):
        # Handling write modes ('w', 'a', etc.)
        if 'w' in mode or 'a' in mode:
            self.modified = True
        if self.tar_file or self.h5_file:
            path = self._rel_path(path)
            if 'w' in mode or 'a' in mode:
                # Creating a memory-like object for writes
                if path not in self.memory_files:
                    self.memory_files[path] = ManagedBytesIO()
                return self.memory_files[path]
            else:
                # Reading from memory
                if path in self.memory_files:
                    memory_file = self.memory_files[path]
                    memory_file.reopen()
                    memory_file.seek(0)
                    return memory_file
                # Reading from the tar or h5 archive
                if self.tar_file:
                    # Read data from tar file
                    return self.tar_file.extractfile(path)
                elif self.h5_file:
                    # Read data from HDF5 file
                    data = self.h5_file[path][:]
                    return io.BytesIO(data.tobytes())
        else:
            return open(path, mode)

    def remove(self, path):
        if self.tar_file or self.h5_file:
            self.modified = True
            path = self._rel_path(path)
            # Mark the file for deletion upon rewriting the tar
            if path in self.memory_files:
                # Explicitly close to prevent leaks
                self.memory_files[path].real_close()
                del self.memory_files[path]
            else:
                self.memory_files[path] = None
        else:
            os.remove(path)

    def writeback(self):
        if self.modified:
            if self.tar_file:
                self.tar_file.close()
                # rename exisiting file to `old`
                old_archive_path = f'{self.archive_path}.old'
                os.rename(self.archive_path, old_archive_path)
                old_tar_file = tarfile.open(old_archive_path, 'r')
                # Create a new tar file with the same name
                with tarfile.open(self.archive_path, 'w:gz' if self.archive_extension.endswith('.gz') else 'w') as new_tar_file:
                    # Re-add unmodified files from the original tar
                    for member in old_tar_file.getmembers():
                        # Check if the file has been removed or replaced
                        if member.name not in self.memory_files:
                            new_tar_file.addfile(
                                member, old_tar_file.extractfile(member.name))

                    # Add or replace modified/new files
                    for path, memory_file in self.memory_files.items():
                        if memory_file is not None:  # Only add if it's not marked as deleted
                            memory_file.seek(0)
                            info = tarfile.TarInfo(name=path)
                            info.size = len(memory_file.getvalue())
                            new_tar_file.addfile(info, memory_file)

                # Remove the old file
                old_tar_file.close()
                os.remove(old_archive_path)
                # update the tar_file pointer
                self.tar_file = tarfile.open(self.archive_path, 'r')
            elif self.h5_file:
                self.h5_file.close()  # Close the current file handle
                # Reopen the file in write mode
                # No need to create a new file as in the tar case
                with h5py.File(self.archive_path, 'a', libver='latest') as hdf:
                    # Write back changes or new data
                    for path, memory_file in self.memory_files.items():
                        # Remove from h5 archive if exists
                        if path in hdf:
                            del hdf[path]
                        # Only add if it's not marked as deleted
                        if memory_file is not None:
                            memory_file.seek(0)
                            data = np.frombuffer(
                                memory_file.read(), dtype='uint8')
                            hdf.create_dataset(path, data=data)
                # Reopen the file in SWMR read mode
                self.h5_file = h5py.File(self.archive_path, 'r', swmr=True)
            # Clear memory storage after writing back
            self.modified = False
            for memory_file in self.memory_files.values():
                if memory_file is not None:
                    memory_file.real_close()
            self.memory_files.clear()

    def refresh(self):
        """Refresh the open file handle to see updated data in SWMR mode."""
        if self.h5_file:
            self.h5_file.refresh()

    def close(self):
        if self.tar_file:
            self.tar_file.close()
        elif self.h5_file:
            self.h5_file.close()

    def __del__(self):
        # Ensure the tar file is closed when the object is deleted
        self.close()


def main():
    import numpy as np
    import pandas as pd

    df = pd.DataFrame(np.random.randint(
        0, 100, size=(100, 4)), columns=list('ABCD'))
    rand_array = np.random.randn(100, 4)
    # Use the real filesystem
    # real_fs = FlexibleFileSystem('demo/8jT9ygmMvMg/scene00001_0')
    # print(real_fs.exists('demo/8jT9ygmMvMg/scene00001_0/colmap'))
    # print(real_fs.listdir('demo/8jT9ygmMvMg/scene00001_0/sparse'))
    # np.savetxt('demo/8jT9ygmMvMg/scene00001_0/np_temp.txt',
    #            rand_array)
    # df.to_csv('demo/8jT9ygmMvMg/scene00001_0/pd_temp.csv')
    # with real_fs.open('demo/8jT9ygmMvMg/scene00001_0/np_temp1.txt', 'w') as f:
    #     np.savetxt(f, rand_array)
    # with real_fs.open('demo/8jT9ygmMvMg/scene00001_0/pd_temp1.csv', 'w') as f:
    #     df.to_csv(f)

    # Mount .tar.gz content into memory and treat it as if it were located under 'virtual_dir'
    memory_fs = FlexibleFileSystem('demo/8jT9ygmMvMg/scene00001_0.tar.gz')
    print(memory_fs.exists('demo/8jT9ygmMvMg/scene00001_0/colmap'))
    print(memory_fs.listdir('demo/8jT9ygmMvMg/scene00001_0/sparse'))
    with memory_fs.open('demo/8jT9ygmMvMg/scene00001_0/np_temp1.txt', 'w') as f:
        np.savetxt(f, rand_array)
    with memory_fs.open('demo/8jT9ygmMvMg/scene00001_0/pd_temp1.csv', 'w') as f:
        df.to_csv(f)
    with memory_fs.open('demo/8jT9ygmMvMg/scene00001_0/pd_temp1.csv', 'r') as f:
        df_ = pd.read_csv(f, index_col=0)
    memory_fs.remove('demo/8jT9ygmMvMg/scene00001_0/pd_temp1.csv')
    memory_fs.writeback()
    with memory_fs.open('demo/8jT9ygmMvMg/scene00001_0/np_temp1.txt', 'r') as f:
        rand_array_ = np.loadtxt(f)
    memory_fs.close()

    def print_all_names_and_items(name, obj):
        print(name, type(obj))

    memory_fs = FlexibleFileSystem('demo/0jC-sW5l4_g/scene00001_0.h5')
    with h5py.File('demo/0jC-sW5l4_g/scene00001_0.h5', 'r') as hdf:
        hdf.visititems(print_all_names_and_items)
        pass
    print(memory_fs.exists('demo/0jC-sW5l4_g/scene00001_0/colmap'))
    print(memory_fs.listdir('demo/0jC-sW5l4_g/scene00001_0/sparse'))
    with memory_fs.open('demo/0jC-sW5l4_g/scene00001_0/np_temp1.txt', 'w') as f:
        np.savetxt(f, rand_array)
    with memory_fs.open('demo/0jC-sW5l4_g/scene00001_0/pd_temp1.csv', 'w') as f:
        df.to_csv(f)
    with memory_fs.open('demo/0jC-sW5l4_g/scene00001_0/pd_temp1.csv', 'r') as f:
        df_ = pd.read_csv(f, index_col=0)
    memory_fs.remove('demo/0jC-sW5l4_g/scene00001_0/pd_temp1.csv')
    memory_fs.remove('demo/0jC-sW5l4_g/scene00001_0/colmap')
    memory_fs.remove('demo/0jC-sW5l4_g/scene00001_0/sparse/1')
    memory_fs.writeback()
    print(memory_fs.listdir('demo/0jC-sW5l4_g/scene00001_0'))
    with memory_fs.open('demo/0jC-sW5l4_g/scene00001_0/np_temp1.txt', 'r') as f:
        rand_array_ = np.loadtxt(f)
    with h5py.File('demo/0jC-sW5l4_g/scene00001_0.h5', 'r') as hdf:
        hdf.visititems(print_all_names_and_items)
        pass

    memory_fs = FlexibleFileSystem(
        'youtube_drone_videos/filter_results_2024-03-30_23-57-41.tar')
    print(len(memory_fs.listdir('youtube_drone_videos')))


if __name__ == '__main__':
    main()
    pass
