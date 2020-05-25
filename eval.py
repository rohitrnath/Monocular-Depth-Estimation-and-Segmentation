from Logger.GridImage import show
from criterion.SSIM import SSIM
import torch
from utils.utils import *


def evaluate(model, device, test_loader):
  criterion_ssim = SSIM()
  val_Acc_Depth = AverageMeter()
  val_Acc_Mask = AverageMeter()
  model.eval()
  N = len(test_loader)
  with torch.no_grad():
    for i, sample_batch in enumerate(test_loader):
      bg_n    = sample_batch['bg'].to(device)
      fg_bg_n = sample_batch['fg_bg'].to(device)
      mask_n  = sample_batch['mask'].to(device)
      depth_n = sample_batch['depth'].to(device)
      
      output = model(bg_n,fg_bg_n)
      
      #Measure Accuracy
      acc_depth = criterion_ssim(output[0], depth_n)
      acc_mask  = criterion_ssim(output[1], mask_n)
      val_Acc_Mask.update(acc_mask, fg_bg_n.size(0))
      val_Acc_Depth.update(acc_depth, fg_bg_n.size(0))

      if i % (N//10) == 0:
        print(f"Sample {i}/{N}")
        print("Input fg_bg")
        show(sample_batch['fg_bg'].cpu(), nrow=10)
        print("Input bg")
        show(sample_batch['bg'].cpu(), nrow=10)
        print("Depth - Groundtruth")
        show(sample_batch['depth'].cpu(), nrow=10)
        print("Depth - Prediction")
        show(output[0].detach().cpu(), nrow=10)
        print("Mask - Groundtruth")
        show(sample_batch['mask'].cpu(), nrow=10)
        print("Mask - Prediction")
        show(output[1].detach().cpu(), nrow=10)

  print(f"Mask  Accuracy : {val_Acc_Mask.avg*100}%")
  print(f"Depth Accuracy : {val_Acc_Depth.avg*100}%")
