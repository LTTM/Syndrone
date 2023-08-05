from utils.data import SelmaDrones
import os
from os import path
import numpy as np
import cv2 as cv
import torch


class Aeroscapes(SelmaDrones):
    def __init__(self,
                 root_path,
                 splits_path,
                 split='train',
                 minlen=0,
                 **kwargs
                 ):
        self.modality = 'rgb'
        self.initconsts()

        assert path.exists(root_path), f"Path [{root_path}] does not exist"
        assert path.isdir(root_path), f"Path [{root_path}] is not a directory"
        self.root_path = root_path


        assert path.exists(splits_path), f"Path [{splits_path}] does not exist"
        assert path.isdir(splits_path), f"Path [{splits_path}] is not a directory"
        split = 'trn' if split=='train' else 'val'
        split_path = path.join(splits_path, split+'.txt')
        assert path.isfile(split_path), f"Split file [{split_path}] does not exist"
        with open(path.join(splits_path, split + '.txt')) as f:
            self.items = [(path.join(self.root_path, "JPEGImages/" + l.strip() + ".jpg"),
                           path.join(self.root_path, "SegmentationClass/" + l.strip() + ".png")) for l in f]
        self.items = self.items * (1 + minlen // len(self.items))

        self.label_names = ["Road", "Nature", "Person", "Vehicle", "Construction", "Obstacle", "Water"]
        self.idmap = {0:-1, 1:2, 2:3, 3:3, 4:-1, 5:-1, 6:-1, 7:5, 8:4, 9:1, 10:0, 11:-1}

    def __getitem__(self, item):
        rgb_path, gt_path = self.items[item]
        rgb = cv.imread(rgb_path, cv.IMREAD_UNCHANGED)[...,
              ::-1] / 255.  # swap to rgb and rescale to 0-1 for further processing
        rgb = (rgb - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        sem = cv.imread(gt_path, cv.IMREAD_UNCHANGED)
        mlb = -1 * np.ones_like(sem, dtype=int)
        for k, v in self.idmap.items():
            mlb[sem == k] = v
        return self.to_pytorch(rgb, None, mlb), ('h', 't')

class ICGDrones(SelmaDrones):
    def __init__(self,
                 root_path,
                 splits_path,
                 split='train',
                 minlen=0,
                 **kwargs
                 ):
        self.modality = 'rgb'
        self.initconsts()

        assert path.exists(root_path), f"Path [{root_path}] does not exist"
        assert path.isdir(root_path), f"Path [{root_path}] is not a directory"
        self.root_path = root_path
        image_paths = []
        images_folder_path = path.join(self.root_path, 'training_set', 'images')
        for image_file in os.listdir(images_folder_path):
            if image_file.lower().endswith('.jpg'):
                image_path = os.path.join(images_folder_path, image_file)
                image_paths.append(image_path)
        label_paths = [s.replace('images', path.join('gt', 'semantic', 'label_images')).replace('jpg', 'png') for s in image_paths]
        self.items = list(zip(image_paths, label_paths))
        self.items = self.items * (1 + minlen // len(self.items))

        self.label_names = ["Road", "Nature", "Person", "Vehicle", "Construction", "Obstacle", "Water"]
        self.idmap = {0: -1, 1: 0, 2: 1, 3: 1, 4: 1, 5: 6, 6: 1, 7: 6, 8: 1, 9: 4, 10: 4, 11: 4, 12: 4,
                      13: 4, 14: 4, 15: 2, 16: -1, 17: 3, 18: 3, 19: 1, 20: 1, 21: -1, 22: 5, 23: -1}

    def encode_segmap(self, segcolors):
        """RGB colors to class labels"""
        colors = np.array([
            [0, 0, 0],  # unlabeled
            [128, 64, 128],  # paved-area
            [130, 76, 0],  # dirt
            [0, 102, 0],  # grass
            [112, 103, 87],  # gravel
            [28, 42, 168],  # water 5
            [48, 41, 30],  # rocks
            [0, 50, 89],  # pool
            [107, 142, 35],  # vegetation 8
            [70, 70, 70],  # roof
            [102, 102, 156],  # wall
            [254, 228, 12],  # window
            [254, 148, 12],  # door
            [190, 153, 153],  # fence
            [153, 153, 153],  # fence-pole
            [255, 22, 96],  # person 15
            [102, 51, 0],  # dog
            [9, 143, 150],  # car 17
            [119, 11, 32],  # bicycle
            [51, 51, 0],  # tree 19
            [190, 250, 190],  # bald-tree
            [112, 150, 146],  # ar-marker
            [2, 135, 115],  # obstacle
            [255, 0, 0]  # conflicting
        ], dtype=np.uint8)
        label_map = colors.shape[0]*np.ones((segcolors.shape[0], segcolors.shape[1]), dtype=np.long)
        for i, color in enumerate(colors):
            label_map[np.all(segcolors == color, axis=2)] = i
        return label_map

    def __getitem__(self, item):
        rgb_path, gt_path = self.items[item]
        rgb = cv.imread(rgb_path, cv.IMREAD_UNCHANGED)[...,
              ::-1] / 255.  # swap to rgb and rescale to 0-1 for further processing
        rgb = (rgb - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        sem = cv.imread(gt_path, cv.IMREAD_UNCHANGED)[..., ::-1]
        sem = self.encode_segmap(sem)
        mlb = -1 * np.ones_like(sem, dtype=int)
        for k, v in self.idmap.items():
            mlb[sem == k] = v
        return self.to_pytorch(rgb, None, mlb), ('h', 't')


class UAVid(SelmaDrones):
    def __init__(self,
                 root_path,
                 splits_path,
                 split='train',
                 minlen=0,
                 **kwargs
                 ):
        self.modality = 'rgb'
        self.initconsts()

        assert path.exists(root_path), f"Path [{root_path}] does not exist"
        assert path.isdir(root_path), f"Path [{root_path}] is not a directory"
        self.root_path = root_path
        split = 'uavid_train' if split == 'train' else 'uavid_val'

        image_paths = []
        folder_path = path.join(self.root_path, split)
        for root, dirs, files in os.walk(folder_path):
            for dir in dirs:
                if 'seq' in dir:
                    images_folder_path = os.path.join(root, dir, 'Images')
                    for image_file in os.listdir(images_folder_path):
                        if image_file.lower().endswith('.png'):
                            image_path = os.path.join(images_folder_path, image_file)
                            image_paths.append(image_path)
        label_paths = [s.replace('Images', 'Labels') for s in image_paths]
        self.items = list(zip(image_paths, label_paths))
        self.items = self.items * (1 + minlen // len(self.items))

        self.label_names = ["Road", "Nature", "Person", "Vehicle", "Construction", "Obstacle", "Water"]
        self.idmap = {0: 4, 1: 0, 2: 3, 3: 1, 4: 1, 5: 3, 6: 2, 7: -1}

    def encode_segmap(self, segcolors):
        """RGB colors to class labels"""
        colors = np.array([
            [128, 0, 0],  # Building
            [128, 64, 128],  # Road
            [192, 0, 192],  # Static car
            [0, 128, 0],  # Tree
            [128, 128, 0],  # Low vegetation
            [64, 0, 128],  # Moving car
            [64, 64, 0],  # Human
            [0, 0, 0]  # Background / clutter
        ], dtype=np.uint8)
        label_map = colors.shape[0]*np.ones((segcolors.shape[0], segcolors.shape[1]), dtype=np.long)
        for i, color in enumerate(colors):
            label_map[np.all(segcolors == color, axis=2)] = i
        return label_map

    def __getitem__(self, item):
        rgb_path, gt_path = self.items[item]
        rgb = cv.imread(rgb_path, cv.IMREAD_UNCHANGED)[...,
              ::-1] / 255.  # swap to rgb and rescale to 0-1 for further processing
        rgb = (rgb - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        sem = cv.imread(gt_path, cv.IMREAD_UNCHANGED)[..., ::-1]
        sem = self.encode_segmap(sem)
        mlb = -1 * np.ones_like(sem, dtype=int)
        for k, v in self.idmap.items():
            mlb[sem == k] = v
        return self.to_pytorch(rgb, None, mlb), ('h', 't')


class UDD5(SelmaDrones):
    def __init__(self,
                 root_path,
                 splits_path,
                 split='train',
                 minlen=0,
                 **kwargs
                 ):
        self.modality = 'rgb'
        self.initconsts()

        assert path.exists(root_path), f"Path [{root_path}] does not exist"
        assert path.isdir(root_path), f"Path [{root_path}] is not a directory"
        self.root_path = root_path

        split = split if split=='train' else 'val'
        split_path = path.join(splits_path, split+'.txt')
        assert path.exists(splits_path), f"Path [{splits_path}] does not exist"
        assert path.isdir(splits_path), f"Path [{splits_path}] is not a directory"
        assert path.isfile(split_path), f"Split file [{split_path}] does not exist"
        with open(path.join(splits_path, split + '.txt')) as f:
            self.items = [(path.join(self.root_path, l.strip().split()[0]),
                           path.join(self.root_path, l.strip().split()[1])) for l in f]
        self.items = self.items * (1 + minlen // len(self.items))

        self.label_names = ["Road", "Nature", "Person", "Vehicle", "Construction", "Obstacle", "Water"]
        self.idmap = {0: 4, 1: 0, 2: 3, 3: 1, 4: -1}

    def encode_segmap(self, segcolors):
        """RGB colors to class labels"""
        colors = np.array([
            [102, 102, 156],  # Building
            [128, 64, 128],  # Road
            [0, 0, 142],  # vehicle
            [107, 142, 35],  # vegetation
            [0, 0, 0]  # background
        ], dtype=np.uint8)
        label_map = colors.shape[0]*np.ones((segcolors.shape[0], segcolors.shape[1]), dtype=np.long)
        for i, color in enumerate(colors):
            label_map[np.all(segcolors == color, axis=2)] = i
        return label_map

    def __getitem__(self, item):
        rgb_path, gt_path = self.items[item]
        rgb = cv.imread(rgb_path, cv.IMREAD_UNCHANGED)[...,
              ::-1] / 255.  # swap to rgb and rescale to 0-1 for further processing
        rgb = (rgb - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        sem = cv.imread(gt_path, cv.IMREAD_UNCHANGED)[..., ::-1]
        sem = self.encode_segmap(sem)
        mlb = -1 * np.ones_like(sem, dtype=int)
        for k, v in self.idmap.items():
            mlb[sem == k] = v
        return self.to_pytorch(rgb, None, mlb), ('h', 't')

class UDD6(SelmaDrones):
    def __init__(self,
                 root_path,
                 splits_path,
                 split='train',
                 minlen=0,
                 **kwargs
                 ):
        self.modality = 'rgb'
        self.initconsts()

        assert path.exists(root_path), f"Path [{root_path}] does not exist"
        assert path.isdir(root_path), f"Path [{root_path}] is not a directory"
        self.root_path = root_path

        split = split if split=='train' else 'val'
        split_path = path.join(splits_path, split+'.txt')
        assert path.exists(splits_path), f"Path [{splits_path}] does not exist"
        assert path.isdir(splits_path), f"Path [{splits_path}] is not a directory"
        assert path.isfile(split_path), f"Split file [{split_path}] does not exist"
        with open(path.join(splits_path, split + '.txt')) as f:
            self.items = [(path.join(self.root_path, l.strip().split()[0]),
                           path.join(self.root_path, l.strip().split()[1])) for l in f]
        self.items = self.items * (1 + minlen // len(self.items))

        self.label_names = ["Road", "Nature", "Person", "Vehicle", "Construction", "Obstacle", "Water"]
        self.idmap = {0: 4, 1: 0, 2: 3, 3: 1, 4: 4, 5: -1}

    def encode_segmap(self, segcolors):
        """RGB colors to class labels"""
        colors = np.array([
            [102, 102, 156],  # Facade
            [128, 64, 128],  # Road
            [0, 0, 142],  # vehicle
            [107, 142, 35],  # vegetation
            [70, 70, 70],    # roof
            [0, 0, 0]  # background
        ], dtype=np.uint8)
        label_map = colors.shape[0]*np.ones((segcolors.shape[0], segcolors.shape[1]), dtype=np.long)
        for i, color in enumerate(colors):
            label_map[np.all(segcolors == color, axis=2)] = i
        return label_map

    def __getitem__(self, item):
        rgb_path, gt_path = self.items[item]
        rgb = cv.imread(rgb_path, cv.IMREAD_UNCHANGED)[...,
              ::-1] / 255.  # swap to rgb and rescale to 0-1 for further processing
        rgb = (rgb - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        sem = cv.imread(gt_path, cv.IMREAD_UNCHANGED)[..., ::-1]
        sem = self.encode_segmap(sem)
        mlb = -1 * np.ones_like(sem, dtype=int)
        for k, v in self.idmap.items():
            mlb[sem == k] = v
        return self.to_pytorch(rgb, None, mlb), ('h', 't')

class VDD(SelmaDrones):
    def __init__(self,
                 root_path,
                 splits_path,
                 split='train',
                 minlen=0,
                 **kwargs
                 ):
        self.modality = 'rgb'
        self.initconsts()

        assert path.exists(root_path), f"Path [{root_path}] does not exist"
        assert path.isdir(root_path), f"Path [{root_path}] is not a directory"
        self.root_path = root_path
        image_paths = []
        images_folder_path = path.join(self.root_path, split, 'src')
        for image_file in os.listdir(images_folder_path):
            if image_file.lower().endswith('.jpg'):
                image_path = os.path.join(images_folder_path, image_file)
                image_paths.append(image_path)
        label_paths = [s.replace('src', 'gt').replace('JPG', 'png') for s in
                       image_paths]
        self.items = list(zip(image_paths, label_paths))
        self.items = self.items * (1 + minlen // len(self.items))

        self.label_names = ["Road", "Nature", "Person", "Vehicle", "Construction", "Obstacle", "Water"]
        self.idmap = {0: 0, 1: 4, 2: -1, 3: 3, 4: 4, 5: 6, 6: -1}
        """
        0: Road
        1: Roof
        
        3: Vegetation
        4: Wall
        
        6: Other
        """

    def __getitem__(self, item):
        rgb_path, gt_path = self.items[item]
        rgb = cv.imread(rgb_path, cv.IMREAD_UNCHANGED)[...,
              ::-1] / 255.  # swap to rgb and rescale to 0-1 for further processing
        rgb = (rgb - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        sem = cv.imread(gt_path, cv.IMREAD_UNCHANGED)
        mlb = -1 * np.ones_like(sem, dtype=int)
        for k, v in self.idmap.items():
            mlb[sem == k] = v
        return self.to_pytorch(rgb, None, mlb), ('h', 't')


if __name__ == "__main__":
    from matplotlib import pyplot as plt


    # d = ICGDrones("../Datasets/ICGDrones",
    #                 "../Datasets/ICGDrones",
    #                 coarselabels=True)

    d = VDD("../Datasets/VDD",
                    "../Datasets/VDD",
                    coarselabels=True,
            split='test')


    for (rgb, mlb), _ in d:
        fig, axs = plt.subplots(1,2)
        axs[0].imshow(d.to_rgb(rgb.permute(1,2,0)))
        axs[1].imshow(mlb) #d.color_label(mlb)
        plt.show()

