import cv2 as cv
from os import path, mkdir
from tqdm import tqdm

import torch
from torch.nn import CrossEntropyLoss, Module, Softmax

from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, deeplabv3_resnet50
from torch.utils.data import DataLoader

from utils.args import get_args
from utils.data import SelmaDrones, coarse_mapping
from utils.data_adaptation import Aeroscapes, ICGDrones, UAVid, UDD5, UDD6, VDD
from utils.metrics import Metrics

class CorsePred(Module):
    def __init__(self, indices, num_classes=7):
        super().__init__()
        self.indices = indices
        self.num_classes = num_classes
        self.soft = Softmax(dim=1)

    def forward(self, out):
        B, C, H, W = out.shape
        sout = self.soft(out)
        cout = torch.zeros(B, self.num_classes, H, W, device=out.device, dtype=out.dtype)
        for b in range(B):
            for i, c in enumerate(self.indices):
                if c >= 0:
                    cout[b, c] += sout[b, i]
        return cout.argmax(dim=1), sout.argmax(dim=1)

if __name__ == "__main__":
    args = get_args()

    if args.adapt_dset == "selmadrones":
        dclass = SelmaDrones
    elif args.adapt_dset == "aeroscapes":
        dclass = Aeroscapes
    elif args.adapt_dset == "icgdrones":
        dclass = ICGDrones
    elif args.adapt_dset == "uavid":
        dclass = UAVid
    elif args.adapt_dset == "udd5":
        dclass = UDD5
    elif args.adapt_dset == "udd6":
        dclass = UDD6
    elif args.adapt_dset == "vdd":
        dclass = VDD


    vset = dclass(root_path=args.root_path,
                       splits_path=args.splits_path,
                       town=args.town,
                       height=args.height,
                       split='train',
                       modality=args.modality,
                       coarselabels=True)
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
    if args.pretrained_ckpt is not None:
        model.load_state_dict(torch.load(args.pretrained_ckpt, map_location='cpu'))
    model.to(device)
    model.eval()

    indices, cnames = coarse_mapping()
    mapper = CorsePred(indices=indices)
    mapper.to(device)

    if args.adapt_dset == "selmadrones":
        settings = [(h,t) for h in vset.height for t in vset.town]
        metrics = {(h,t): Metrics(cnames, device=device, log_colors=False) for h,t in settings}
        hmetrics = {h: Metrics(cnames, device=device, log_colors=False) for h in vset.height}
        tmetrics = {t: Metrics(cnames, device=device, log_colors=False) for t in vset.town}
    allmetric = Metrics(cnames, device=device)
    if not path.exists(args.evaldir):
        mkdir(args.evaldir)
    if not path.exists(path.join(args.evaldir, "cpred")):
        mkdir(path.join(args.evaldir, "cpred"))
        mkdir(path.join(args.evaldir, "fpred"))
    
    with torch.no_grad():
        for ii, ((rgb, mlb), (h,t)) in enumerate(tqdm(vloader, total=len(vloader), desc="Testing...")):
            rgb, mlb = rgb.to(device, dtype=torch.float32), mlb.to(device, dtype=torch.long)
            h, t = h[0], t[0]

            out = model(rgb)['out']
            cpred, fpred = mapper(out) #out.argmax(dim=1)

            cv.imwrite(path.join(args.evaldir, "cpred", "%05d.png"%ii), vset.ccmap[cpred[0].cpu().numpy()][...,::-1])
            cv.imwrite(path.join(args.evaldir, "fpred", "%05d.png" % ii), vset.cmap[fpred[0].cpu().numpy()][..., ::-1])

            if args.adapt_dset == "selmadrones":
                metrics[(h,t)].add_sample(cpred, mlb)
            else:
                allmetric.add_sample(cpred, mlb)
            #break

    if args.adapt_dset == "selmadrones":
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
