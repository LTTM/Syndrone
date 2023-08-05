import cv2 as cv
import numpy as np
from os import path
import torch
from torch.utils.data import Dataset

class SelmaDrones(Dataset):
    def __init__(self,
                 root_path,
                 splits_path,
                 town='all',
                 seqnlen='120',
                 scenario='ClearNoon',
                 height='all',
                 split='train',
                 modality='rgb',
                 minlen=0):

        assert path.exists(root_path), f"Path [{root_path}] does not exist"
        assert path.isdir(root_path), f"Path [{root_path}] is not a directory"
        self.root_path = root_path

        assert path.exists(splits_path), f"Path [{splits_path}] does not exist"
        assert path.isdir(splits_path), f"Path [{splits_path}] is not a directory"
        split_path = path.join(splits_path, split+'.txt')
        assert path.isfile(split_path), f"Split file [{split_path}] does not exist"
        idxs = ["%05d"%int(l.strip()) for l in open(split_path, 'r')]
        
        assert town in ['01', '02', '03', '04', '05', '06', '07', '10HD', 'all'], f"Illegal Value [{town}] for 'town' argument"
        town = ['01', '02', '03', '04', '05', '06', '07', '10HD'] if town == 'all' else [town]

        assert height in ['20', '50', '80', 'all'], f"Illegal Value [{height}] for 'height' argument"
        height = ['20', '50', '80'] if height == 'all' else [height]

        assert all([path.isdir(path.join(root_path, "Town"+t+"_Opt_"+seqnlen, scenario, "height"+h+"m")) for t in town for h in height]), \
            f"Missing some town/height combinations for [{scenario}] scenario"

        assert modality in ['depth', 'rgb', 'all'], f"Illegal Value [{modality}] for 'modality' argument"
        self.modality = modality

        self.town = town
        self.height = height

        self.items = [(path.join(root_path, "Town"+t+"_Opt_"+seqnlen, scenario, "height"+h+"m", "%s", i+".%s"), (h,t)) for t in town for h in height for i in idxs]
        self.items = self.items * (1 + minlen//len(self.items))

        self.label_names = [
                                "Building",
                                "Fence",
                                "Other",
                                "Pole",
                                "RoadLine",
                                "Road",
                                "Sidewalk",
                                "Vegetation",
                                "Wall",
                                "Traffic Signs",
                                "Sky",
                                "Ground",
                                "Bridge",
                                "Rail Track",
                                "Guard Rail",
                                "Traffic Light",
                                "Static",
                                "Dynamic",
                                "Water",
                                "Terrain",
                                "Person",
                                "Rider",
                                "Car",
                                "Truck",
                                "Bus",
                                "Train",
                                "Motorcycle",
                                "Bicycle"
                            ]
        self.idmap = {1:0, 2:1, 3:2, 5:3, 6:4, 7:5, 8:6, 9:7, 11:8, 12:9, 13:10, 14:11, 15:12, 16:13, 17:14,
                      18:15, 19:16, 20:17, 21:18, 22:19, 40:20, 41:21, 100:22, 101:23, 102:24, 103:25, 104:26, 105:27}
        self.cmap = np.array([
            [ 70, 70, 70], # building
            [190,153,153], # fence
            [180,220,135], # other
            [153,153,153], # pole
            [255,255,255], # road line
            [128, 64,128], # road
            [244, 35,232], # sidewalk
            [107,142, 35], # vegetation
            [102,102,156], # wall
            [220,220,  0], # traffic sign
            [ 70,130,180], # sky
            [ 81,  0, 81], # ground
            [150,100,100], # bridge
            [230,150,140], # rail track
            [180,165,180], # guard rail
            [250,170, 30], # traffic light
            [110,190,160], # static
            [111, 74,  0], # dynamic
            [ 45, 60,150], # water
            [152,251,152], # terrain
            [220, 20, 60], # person
            [255,  0,  0], # rider
            [  0,  0,142], # car
            [  0,  0, 70], # truck
            [  0, 60,100], # bus
            [  0, 80,100], # train
            [  0,  0,230], # motorcycle
            [119, 11, 32], # bicycle
            [  0,  0,  0], # unknown
        ], dtype=np.uint8)
        
    def __getitem__(self, item):
        fpath, (h,t) = self.items[item]

        if self.modality in ['rgb', 'all']:
            rgb = cv.imread(fpath%('rgb', 'jpg'), cv.IMREAD_UNCHANGED)[...,::-1]/255. # swap to rgb and rescale to 0-1 for further processing
            rgb = (rgb - [0.485, 0.456, 0.406])/[0.229, 0.224, 0.225]
        else:
            rgb = None

        if self.modality in ['depth', 'all']:
            #dth = 1000. * cv.imread(fpath%('depth', 'png'), cv.IMREAD_UNCHANGED)/(256 * 256 - 1) # depth in meters
            dth = cv.imread(fpath%('depth', 'png'), cv.IMREAD_UNCHANGED)/(256 * 256 - 1)
            dth = cv.merge([dth,dth,dth])
            dth = (dth - [0.485, 0.456, 0.406])/[0.229, 0.224, 0.225]
            #dth[dth < 1000.] -= int(h) # normalize w.r.t. the drone height the valid depths
        else:
            dth = None

        sem = cv.imread(fpath%('semantic', 'png'), cv.IMREAD_UNCHANGED)
        mlb = -1*np.ones_like(sem, dtype=int)
        for k,v in self.idmap.items():
            mlb[sem==k] = v
        
        return self.to_pytorch(rgb, dth, mlb), (h,t)

    def __len__(self):
        return len(self.items)

    def to_rgb(self, ts):
        return np.array(ts)*[0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    
    def color_label(self, ts):
        return self.cmap[np.array(ts)]
    
    def scale_depth(self, ts):
        ts[ts<0] = -np.log10(-ts[ts<0])
        ts[ts>0] = np.log10(ts[ts>0])
        return ts
    
    def to_pytorch(self, rgb, dth, mlb):
        if self.modality == 'all':
            return torch.from_numpy(rgb).permute(2,0,1), torch.from_numpy(dth).permute(2,0,1), torch.from_numpy(mlb)
        if self.modality == 'rgb':
            return torch.from_numpy(rgb).permute(2,0,1), torch.from_numpy(mlb)
        return torch.from_numpy(dth).permute(2,0,1), torch.from_numpy(mlb)

if __name__ == "__main__":
    from matplotlib import pyplot as plt

    d = SelmaDrones("D:/selmadrones/renders",
                    "D:/selmadrones",
                    modality='depth')
    
    for (rgb, mlb), _ in d:
        fig, axs = plt.subplots(1,2)
        axs[0].imshow(d.to_rgb(rgb.permute(1,2,0)))
        axs[1].imshow(d.color_label(mlb))
        plt.show()