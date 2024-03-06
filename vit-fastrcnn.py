import torch
import torch.nn as nn
import torchvision
from vit import ViT
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator


def create_model(num_classes):
   backbone = ViT(
      image_size = 512,
      patch_size = 32,
      num_classes = 1000,
      dim = 1024,
      depth = 6,
      heads = 16,
      mlp_dim = 2048,
      dropout = 0.1,
      emb_dropout = 0.1,
   )
   backbone.out_channels = 256
   anchor_generator = AnchorGenerator(
      sizes=((16, 32, 64),),
      aspect_ratios=((0.5, 1.0, 2.0),)
   )
   
   roi_pooler = torchvision.ops.MultiScaleRoIAlign(
      featmap_names=['0'],
      output_size=7,
      sampling_ratio=2
   )
   
   model = FasterRCNN(
      backbone=backbone,
      num_classes=num_classes,
      rpn_anchor_generator=anchor_generator,
      box_roi_pool=roi_pooler,
      min_size=512
   )
   # print(model)
   return model
   
if __name__ == '__main__':
   x = torch.rand(2, 3, 512, 512)
   model = create_model(3)
   model.eval()
   with torch.no_grad():
      output = model(x)
   print(output)