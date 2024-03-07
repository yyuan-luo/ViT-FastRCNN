import torch
import torchvision
from vit import ViTFeatureExtractor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator


def create_model(num_classes, image_size=800, path_size=32):
   backbone = ViTFeatureExtractor(
      image_size = 800,
      patch_size = path_size,
      dim = 1024,
      depth = 6,
      heads = 16,
      mlp_dim = 2048,
      dropout = 0.1,
      emb_dropout = 0.1,
   )
   backbone.out_channels = 512
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
   )
   return model
   
if __name__ == '__main__':
   x = torch.rand(2, 3, 512, 512)
   model = create_model(3)
   model.eval()
   with torch.no_grad():
      output = model(x)
   print(output)