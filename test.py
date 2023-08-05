import numpy as np
from os import path, mkdir
from shutil import rmtree
from tensorboardX import SummaryWriter
from tqdm import tqdm

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, deeplabv3_resnet50
from torch.utils.data import DataLoader

from utils.args import get_args
from utils.data import SelmaDrones
from utils.metrics import Metrics

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
                         num_workers=args.dloader_workers)

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    if args.model == 'mobilenet':
        model = deeplabv3_mobilenet_v3_large(num_classes=28) # uses stride 16 (https://pytorch.org/vision/main/_modules/torchvision/models/segmentation/deeplabv3.html)
    else:
        model = deeplabv3_resnet50(num_classes=28)
    
    model.load_state_dict(torch.load(args.pretrained_ckpt, map_location='cpu'))
    model.to(device)
    model.eval()
    
    settings = [(h,t) for h in vset.height for t in vset.town]

    metrics = {(h,t): Metrics(vset.label_names, device=device, log_colors=False) for h,t in settings}
    hmetrics = {h: Metrics(vset.label_names, device=device, log_colors=False) for h in vset.height}
    tmetrics = {t: Metrics(vset.label_names, device=device, log_colors=False) for t in vset.town}
    allmetric = Metrics(vset.label_names, device=device)
    mkdir(args.evaldir)
    
    with torch.no_grad():
        for (rgb, mlb), (h,t) in tqdm(vloader, total=len(vloader), desc="Testing..."):
            rgb, mlb = rgb.to(device, dtype=torch.float32), mlb.to(device, dtype=torch.long)
            h, t = h[0], t[0]

            out = model(rgb)['out']
            pred = out.argmax(dim=1)
            metrics[(h,t)].add_sample(pred, mlb)
            #break

    for h, t in settings:
        with open(args.evaldir+"/h%s_t%s.txt"%(h,t), "w") as fout:
            fout.write(str(metrics[(h,t)]))
        allmetric.confusion_matrix += metrics[(h,t)].confusion_matrix
        tmetrics[t].confusion_matrix += metrics[(h,t)].confusion_matrix
        hmetrics[h].confusion_matrix += metrics[(h,t)].confusion_matrix

    for h in vset.height:
        with open(args.evaldir+"/h%s.txt"%h, "w") as fout:
            fout.write(str(hmetrics[h]))
    
    for t in vset.town:
        with open(args.evaldir+"/t%s.txt"%t, "w") as fout:
            fout.write(str(tmetrics[t]))

    print(allmetric)
    allmetric.log_colors = False
    with open(args.evaldir+"/all.txt", "w") as fout:
        fout.write(str(allmetric))
