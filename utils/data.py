import cv2 as cv
import json
import numpy as np
from os import path
from scipy.spatial.transform import Rotation as R

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

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

        self.K = np.array([
            [  0, 960, 960],
            [960,   0, 540],
            [  0,   0,   1]
        ])

        self.label_names = [
                            "Person",
                            "Car",
                            "Truck",
                            "Bus",
                            "Train",
                            "Motorcycle",
                            "Bicycle",
                            "Motorcyclist",
                            "Bicyclist"
                            ]
        self.idmap = {(40,):1, (100,):2, (101,):3, (102,):4, (103,):5, (104,):6, (105,):7, (104, 41):8, (105, 41):9, (41, 104):8, (41, 105):9}
        self.cmap = np.array([
            [220, 20, 60], # person
            [  0,  0,142], # car
            [  0,  0, 70], # truck
            [  0, 60,100], # bus
            [  0, 80,100], # train
            [  0,  0,230], # motorcycle
            [119, 11, 32], # bicycle
            [156, 14,168], # motorcyclist
            [177, 16, 48]  # bicyclist
        ], dtype=np.uint8)
        
    def __getitem__(self, item):
        fpath, (h,t) = self.items[item]

        rgb = cv.imread(fpath%('rgb', 'jpg'), cv.IMREAD_UNCHANGED)[...,::-1]/255.
        #rgb = (rgb - [0.485, 0.456, 0.406])/[0.229, 0.224, 0.225]
        
        sem = cv.imread(fpath%('semantic', 'png'), cv.IMREAD_UNCHANGED)
        dth = 1000. * cv.imread(fpath%('depth', 'png'), cv.IMREAD_UNCHANGED)/(256 * 256 - 1)
        
        camera = json.load(open(fpath%('camera', 'json')))
        shift = np.array([camera['x'], camera['y'], camera['z']])
        rotation = R.from_euler('yzx', [90-camera['pitch'], camera['yaw'], camera['roll']], degrees=True).as_matrix()
        
        bpath = fpath.replace('\\', '/').replace('height'+h+'m/', '')%('bboxes', 'json')
        bboxes = json.load(open(bpath))

        bbs = np.array([bb['corners'] for bb in bboxes]) - shift
        bbs = bbs @ rotation
        pbb = bbs @ self.K.T
        valid = np.any(pbb[...,-1] > 0, axis=-1)

        pbb /= pbb[...,-1:] + 1e-5
        uls = pbb.min(axis=1)
        lrs = pbb.max(axis=1)

        vboxes = []
        for v, ul, lr, bb in zip(valid, uls, lrs, bboxes):
            if v:
                x0, y0 = np.round(ul).astype(int)[:2]
                x1, y1 = np.round(lr).astype(int)[:2]
                x0 = np.clip(x0, a_min=0, a_max=1920)
                x1 = np.clip(x1, a_min=0, a_max=1920)
                y0 = np.clip(y0, a_min=0, a_max=1080)
                y1 = np.clip(y1, a_min=0, a_max=1080)

                if x1 > x0 and y1 > y0 and (x1-x0)*(y1-y0) < 1920*1080/2:
                    roi = sem[y0:y1, x0:x1]
                    flag = False
                    for cl in bb['class']:
                        flag = flag or np.any(roi == cl)
                    if flag:
                        vboxes.append(([x0, y0, x1, y1], self.idmap[tuple(bb['class'])]))

        return self.to_pytorch(rgb, vboxes, dth), (h,t)

    def __len__(self):
        return len(self.items)

    def to_rgb(self, ts):
        return np.array(ts) #*[0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    
    def get_figure(self, rgb, bboxes, cls):
        with torch.no_grad():
            fig, ax = plt.subplots()
            ax.imshow(rgb.permute(1,2,0).cpu().numpy())
            for bbox, cl in zip(bboxes, cls):
                x0, y0, x1, y1 = bbox.cpu().numpy()
                r = Rectangle([x0, y0], x1-x0, y1-y0, alpha=.5, color=self.cmap[cl]/255.)
                ax.add_patch(r)
            plt.gca().set_axis_off()
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                        hspace = 0, wspace = 0)
            plt.margins(0,0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
        return fig
    
    def to_pytorch(self, rgb, boxes, dth):
        rgb = torch.from_numpy(rgb).permute(2,0,1).to(dtype=torch.float32)
        bbs = torch.tensor([bb[0] for bb in boxes], dtype=torch.float32)
        cls = torch.tensor([bb[1] for bb in boxes], dtype=torch.long)
        return rgb, bbs, cls, dth

if __name__ == "__main__":
    d = SelmaDrones("D:/selmadrones/renders",
                    "D:/selmadrones")

    for (rgb, bboxes, cls), _ in d:
        fig, axs = plt.subplots(1,2)
        axs[0].imshow(d.to_rgb(rgb.permute(1,2,0)))
        axs[1].imshow(d.to_rgb(rgb.permute(1,2,0)))
        for bbox, cl in zip(bboxes, cls):
            x0, y0, x1, y1 = bbox
            r = Rectangle([x0, y0], x1-x0, y1-y0, alpha=.5, color=d.cmap[cl]/255.)
            axs[1].add_patch(r)
        fig.suptitle(cls)
        plt.show()
