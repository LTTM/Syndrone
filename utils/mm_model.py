import torch
from torch import nn
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

class LateFuse(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.rgb = deeplabv3_mobilenet_v3_large(num_classes=num_classes)
        self.dth = deeplabv3_mobilenet_v3_large(num_classes=num_classes)
        self.merge = nn.Conv2d(2*num_classes, num_classes, 1, bias=False)

    def forward(self, c, d):
        c = self.rgb(c)['out']
        d = self.dth(d)['out']
        x = torch.cat([c,d], dim=1)
        x = self.merge(x)
        return {'out': x}
    
class EarlyFuse(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.dlv3 = deeplabv3_mobilenet_v3_large(num_classes=num_classes)
        with torch.no_grad():
            Co, Ci, K1, K2 = self.dlv3.backbone['0'][0].weight.shape
            w = torch.empty(Co, 4, K1, K2)
            nn.init.xavier_uniform_(w)
            self.dlv3.backbone['0'][0].weight = nn.Parameter(w)

    def forward(self, x):
        return self.dlv3(x)
    
if __name__ == "__main__":
    EarlyFuse(28)