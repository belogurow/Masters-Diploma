from os import listdir
from os.path import join, isfile

import ImageUtils
from ImageIterator import ImageIterator


class ImageFolderIterator(ImageIterator):
    def __init__(self, image_folder_path, interval):
        self.image_folder_path = image_folder_path
        self.interval = interval

        self.image_pathes = []
        self.idx = 0

    def find_files(self):
        for file_name in sorted(listdir(self.image_folder_path)):
            full_file_name = join(self.image_folder_path, file_name)

            if isfile(full_file_name) and ImageUtils.is_image(full_file_name):
                self.image_pathes.append(full_file_name)

    def get_next_image(self):
        self.idx += 1
        return self.image_pathes[self.idx - 1]
