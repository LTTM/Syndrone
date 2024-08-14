import numpy as np
from os import path
from shutil import rmtree
from tensorboardX import SummaryWriter
from tqdm import tqdm

import torch
from torch.optim import Adam
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torch.utils.data import DataLoader

from torchmetrics.detection.mean_ap import MeanAveragePrecision

from utils.args import get_args
from utils.data import SelmaDrones

def cosinescheduler(it, niters, baselr=2.5e-4, warmup=2000):
    if it <= warmup:
        return baselr*it/warmup
    it -= warmup
    scale = np.cos((it/(niters-warmup))*(np.pi/2))**2
    return scale*baselr

def collatefn(data):
    ins, tgs, hs, ts = [], [], [], []
    for batch in data:
        (rgb, bboxes, cls, _), (h, t) = batch
        ins.append(rgb)
        tgs.append({'boxes': bboxes, 'labels': cls})
        hs.append(h)
        ts.append(t)
    return (ins, tgs), (hs, ts)

if __name__ == "__main__":
    args = get_args()

    tset = SelmaDrones(root_path=args.root_path,
                       splits_path=args.splits_path,
                       town=args.town,
                       height=args.height,
                       split='train',
                       modality=args.modality,
                       minlen=args.iters_per_epoch)
    tloader = DataLoader(tset,
                         batch_size=args.batch_size,
                         shuffle=True,
                         pin_memory=True,
                         drop_last=True,
                         num_workers=args.dloader_workers,
                         collate_fn=collatefn)

    vset = SelmaDrones(root_path=args.root_path,
                       splits_path=args.splits_path,
                       town=args.town,
                       height=args.height,
                       split='test',
                       modality=args.modality,
                       minlen=0)
    vloader = DataLoader(vset,
                         batch_size=args.batch_size,
                         shuffle=False,
                         pin_memory=True,
                         drop_last=False,
                         num_workers=args.dloader_workers,
                         collate_fn=collatefn)

    if args.override_logs:
        rmtree(args.logdir, ignore_errors=True)
    if path.exists(args.logdir):
        raise ValueError("Loggin Directory Exists, Stopping. If you want to override it turn on the [override_logs] flag.")
    
    writer = SummaryWriter(args.logdir, flush_secs=.5)
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model = fasterrcnn_mobilenet_v3_large_fpn(num_classes=10)
    model.to(device)
    optim = Adam(model.parameters(), lr=args.lr)

    it = 0
    for e in range(args.epochs):
        model.train()
        for ii, ((ins, tgs), _) in enumerate(tqdm(tloader, total=args.iters_per_epoch, desc="Training Epoch %d/%d"%(e+1, args.epochs))):
            optim.zero_grad()
            lr = cosinescheduler(it, args.epochs*args.iters_per_epoch, args.lr, warmup=args.warmup_iters)
            optim.param_groups[0]['lr'] = lr

            ins, tgs = [rgb.to('cuda') for rgb in ins], \
                            [{k: v.to('cuda') for k,v in tg.items()} for tg in tgs]
            
            if any(len(tg['boxes']) == 0 for tg in tgs):
                continue

            out = model(ins, tgs)
            l = out['loss_classifier'] + out['loss_box_reg'] + out['loss_objectness'] + out['loss_rpn_box_reg']
            l.backward()

            writer.add_scalar('train/lr', lr, it)
            writer.add_scalar('train/loss', l.item(), it)
            writer.add_scalars('train/losses', {k: v.item() for k,v in out.items()}, it)

            optim.step()
            it += 1
            if ii >= args.iters_per_epoch:
                break
            if args.debug:
                break

        writer.add_figure('train/label', tset.get_figure(ins[0], tgs[0]['boxes'], tgs[0]['labels']), it, close=True)
        torch.save(model.state_dict(), args.logdir+"/latest.pth")

        model.eval()
        metrics = MeanAveragePrecision()
        with torch.no_grad():
            for (ins, tgs), _ in tqdm(vloader, total=len(vloader), desc="Test Epoch %d/%d"%(e+1, args.epochs)):                
                ins, tgs = [rgb.to('cuda') for rgb in ins], \
                            [{k: v.to('cuda') for k,v in tg.items()} for tg in tgs]
            
                if any(len(tg['boxes']) == 0 for tg in tgs):
                    continue

                out = model(ins)
                metrics(out, tgs)

                if args.debug:
                    break

        mout = metrics.compute()
        writer.add_scalar('val/mAP@50', mout['map_50'], it)
        writer.add_scalar('val/mAP@75', mout['map_75'], it)
        writer.add_figure('val/label', tset.get_figure(ins[0], tgs[0]['boxes'], tgs[0]['labels']), it, close=True)
        writer.add_figure('val/pred', tset.get_figure(ins[0], out[0]['boxes'], out[0]['labels']), it, close=True)
