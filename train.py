import numpy as np
from os import path
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

def cosinescheduler(it, niters, baselr=2.5e-4, warmup=2000):
    if it <= warmup:
        return baselr*it/warmup
    it -= warmup
    scale = np.cos((it/(niters-warmup))*(np.pi/2))**2
    return scale*baselr

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
                         num_workers=args.dloader_workers)

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
                         num_workers=args.dloader_workers)

    if args.override_logs:
        rmtree(args.logdir, ignore_errors=True)
    if path.exists(args.logdir):
        raise ValueError("Loggin Directory Exists, Stopping. If you want to override it turn on the [override_logs] flag.")
    
    writer = SummaryWriter(args.logdir, flush_secs=.5)
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    if args.model == 'mobilenet':
        model = deeplabv3_mobilenet_v3_large(num_classes=28) # uses stride 16 (https://pytorch.org/vision/main/_modules/torchvision/models/segmentation/deeplabv3.html)
    else:
        model = deeplabv3_resnet50(num_classes=28)
    model.to(device)

    loss = CrossEntropyLoss(ignore_index=-1)
    loss.to(device)

    optim = Adam(model.parameters(), lr=args.lr)

    it = 0
    for e in range(args.epochs):
        model.train()
        metrics = Metrics(tset.label_names, device=device)
        for ii, ((rgb, mlb), _) in enumerate(tqdm(tloader, total=args.iters_per_epoch, desc="Training Epoch %d/%d"%(e+1, args.epochs))):
            optim.zero_grad()

            lr = cosinescheduler(it, args.epochs*args.iters_per_epoch, args.lr, warmup=args.warmup_iters)
            optim.param_groups[0]['lr'] = lr

            rgb, mlb = rgb.to(device, dtype=torch.float32), mlb.to(device, dtype=torch.long)

            out = model(rgb)['out']
            l = loss(out, mlb)
            l.backward()

            pred = out.detach().argmax(dim=1)
            metrics.add_sample(pred, mlb)

            writer.add_scalar('train/lr', lr, it)
            writer.add_scalar('train/loss', l.item(), it)
            writer.add_scalar('train/mIoU', metrics.percent_mIoU(), it)

            optim.step()
            it += 1
            if ii >= args.iters_per_epoch:
                break
            if args.debug:
                break
        writer.add_image('train/input', tset.to_rgb(rgb[0].permute(1,2,0).cpu()), it, dataformats='HWC')
        writer.add_image('train/label', tset.color_label(mlb[0].cpu()), it, dataformats='HWC')
        writer.add_image('train/pred', tset.color_label(pred[0].cpu()), it, dataformats='HWC')
        torch.save(model.state_dict(), args.logdir+"/latest.pth")

        model.eval()
        metrics = Metrics(tset.label_names, device=device)
        with torch.no_grad():
            for (rgb, mlb), _ in tqdm(vloader, total=len(vloader), desc="Test Epoch %d/%d"%(e+1, args.epochs)):
                rgb, mlb = rgb.to(device, dtype=torch.float32), mlb.to(device, dtype=torch.long)

                out = model(rgb)['out']
                pred = out.argmax(dim=1)
                metrics.add_sample(pred, mlb)
                if args.debug:
                    break

            writer.add_scalar('test/mIoU', metrics.percent_mIoU(), it)
            writer.add_image('test/input', tset.to_rgb(rgb[0].permute(1,2,0).cpu()), it, dataformats='HWC')
            writer.add_image('test/label', tset.color_label(mlb[0].cpu()), it, dataformats='HWC')
            writer.add_image('test/pred', tset.color_label(pred[0].cpu()), it, dataformats='HWC')
            print(metrics)