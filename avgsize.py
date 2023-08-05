from matplotlib import pyplot as plt
import numpy as np
import pickle
from os import path, mkdir
from tqdm import tqdm

import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torch.utils.data import DataLoader

from utils.args import get_args
from utils.data import SelmaDrones

def collatefn(data):
    ins, tgs, hs, ts = [], [], [], []
    dts = []
    for batch in data:
        (rgb, bboxes, cls, dth), (h, t) = batch
        ins.append(rgb)
        dts.append(dth)
        tgs.append({'boxes': bboxes, 'labels': cls})
        hs.append(h)
        ts.append(t)
    return (ins, tgs, dts), (hs, ts)

if __name__ == "__main__":
    args = get_args()

    vset = SelmaDrones(root_path=args.root_path,
                       splits_path=args.splits_path,
                       town=args.town,
                       height=args.height,
                       split='test',
                       modality=args.modality)
    vloader = DataLoader(vset,
                         batch_size=1,
                         shuffle=False,
                         pin_memory=True,
                         drop_last=False,
                         num_workers=args.dloader_workers,
                         collate_fn=collatefn)

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    if not path.exists(args.evaldir):
        mkdir(args.evaldir)
    
    data = {i:[] for i in range(9)}
    with torch.no_grad():
        for (ins, tgs, dth), _ in tqdm(vloader, total=len(vloader), desc="Testing..."):                
            if any(len(tg['boxes']) == 0 for tg in tgs):
                continue

            for b, l in zip(tgs[0]['boxes'], tgs[0]['labels']):
                x0, y0, x1, y1 = b.to(dtype=int)
                area = (x1-x0)*(y1-y0)
                d = dth[0][y0:y1,x0:x1].mean().item()
                data[l.item()].append([area.item(), d])
    
    for k in data:
        data[k] = np.array(data[k])

    with open(path.join(args.evaldir,"gts.pkl"), 'wb') as fout:
        pickle.dump(data, fout)