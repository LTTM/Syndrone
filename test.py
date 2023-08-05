from matplotlib import pyplot as plt
import numpy as np
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
    
    model = fasterrcnn_mobilenet_v3_large_fpn(num_classes=9)
    model.load_state_dict(torch.load(args.pretrained_ckpt, map_location='cpu'))
    model.to(device)
    model.eval()
    
    if not path.exists(args.evaldir):
        mkdir(args.evaldir)
    
    data = []
    with torch.no_grad():
        for (ins, tgs, dth), _ in tqdm(vloader, total=len(vloader), desc="Testing..."):                
            ins, tgs = [rgb.to('cuda') for rgb in ins], \
                        [{k: v.to('cuda') for k,v in tg.items()} for tg in tgs]
        
            if any(len(tg['boxes']) == 0 for tg in tgs):
                continue

            out = model(ins)
            #iss = {'boxes': [], 'labels': [], 'iou': []}
            for b, l in zip(out[0]['boxes'], out[0]['labels']):
                x0, y0, x1, y1 = b
                area = (x1-x0)*(y1-y0)
                for tb, tl in zip(tgs[0]['boxes'], tgs[0]['labels']):
                    tx0, ty0, tx1, ty1 = tb
                    tarea = (tx1-tx0)*(ty1-ty0)
                    ix0, iy0, ix1, iy1 = max(x0, tx0), max(y0, ty0), min(x1, tx1), min(y1, ty1)
                    iarea = (ix1-ix0)*(iy1-iy0) if ix1 > ix0 and iy1 > iy0 else 0
                    if iarea > 1 and l == tl:
                        #iss['boxes'].append(torch.tensor([ix0, iy0, ix1, iy1]))
                        #iss['labels'].append(l)
                        #iss['iou'].append(iarea/(area+tarea-iarea))
                        dm = dth[0][int(torch.round(iy0)):int(torch.round(iy1)), \
                                    int(torch.round(ix0)):int(torch.round(ix1))]
                        data.append([(iarea/(area+tarea-iarea)).item(), iarea.item(), dm.mean().item()])
            #print(iss['iou'])
            #vset.get_figure(ins[0], iss['boxes'], iss['labels'])
            #plt.show()
            #break
    
    np.save(path.join(args.evaldir,"bb_ious.npy"), np.array(data))