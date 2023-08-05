from utils.mm_model import EarlyFuse, LateFuse
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, deeplabv3_resnet50
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, retinanet_resnet50_fpn

import torch
from ptflops import get_model_complexity_info

if __name__ == "__main__":

    with torch.no_grad():
        mnet = deeplabv3_mobilenet_v3_large(num_classes=28).eval()
        rn50 = deeplabv3_resnet50(num_classes=28).eval()
        mmef = EarlyFuse(num_classes=28).eval()
        mmlf = LateFuse(num_classes=28).eval()
        frcn = fasterrcnn_mobilenet_v3_large_fpn(num_classes=9).eval()
        rnet = retinanet_resnet50_fpn(num_classes=9).eval()

        mnet_macs, _ = get_model_complexity_info(mnet, (3, 1920, 1080), as_strings=True,
                                                    print_per_layer_stat=False, verbose=False)
        
        rn50_macs, _ = get_model_complexity_info(rn50, (3, 1920, 1080), as_strings=True,
                                                    print_per_layer_stat=False, verbose=False)
        
        mmef_macs, _ = get_model_complexity_info(mmef, (4, 1920, 1080), as_strings=True,
                                                    print_per_layer_stat=False, verbose=False)

        mmlf_macs, _ = get_model_complexity_info(mmlf, (3, 1920, 1080), as_strings=True, input_constructor=lambda x: {'c': torch.zeros(1, *x), 'd': torch.zeros(1, *x)},
                                                    print_per_layer_stat=False, verbose=False)
        
        frcn_macs, _ = get_model_complexity_info(frcn, (3, 1920, 1080), as_strings=True,
                                                    print_per_layer_stat=False, verbose=False)
        
        rnet_macs, _ = get_model_complexity_info(rnet, (3, 1920, 1080), as_strings=True,
                                                    print_per_layer_stat=False, verbose=False)
        print(mnet_macs, rn50_macs, mmef_macs, mmlf_macs, frcn_macs, rnet_macs)