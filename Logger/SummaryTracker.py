import os
import torch
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
import torchvision
import matplotlib.pyplot as plt

class SummaryTracker(object):
    def __init__(self, directory):
        self._base_dir = directory
        self._tensorboard_dir = os.path.join(self._base_dir, "tensorboard/")
        self._checkpoints_dir = os.path.join(self._base_dir, "checkpoints/")
        # self._plots_dir = os.path.join(self._base_dir, "plots/")
        # self._val_disp_dir = os.path.join(self._plots_dir, "validation-disparities/")
        # Store best loss for model checkpoints
        self._best_val_loss = float("inf")
        self._best_val_acc_mask =  0
        self._best_val_acc_depth=  0
        self._best_mask_cpt_path = os.path.join(self._checkpoints_dir, "best-mask-model.pth")
        self._best_depth_cpt_path = os.path.join(self._checkpoints_dir, "best-depth-model.pth")
        self._last_cpt_path = os.path.join(self._checkpoints_dir, "last-model.pth")
        self._create_dirs()
        self.create_summary()


    def addGraph(self, model, batchSize):
        dummy_input = (torch.zeros(batchSize, 3, 224, 224),torch.zeros(batchSize, 3, 224, 224))
        writer = SummaryWriter(os.path.join(self._tensorboard_dir))
        writer.add_graph( model, dummy_input)
        writer.close()
        del dummy_input

    def create_summary(self):
        self.writer = SummaryWriter(log_dir=self._tensorboard_dir)
        #return writer

    def addToSummary(self, name, item, global_step):
      self.writer.add_scalar( name, item, global_step)

    def visualize_image(self, cat, dataset, pred, global_step):

        grid_image = self.show( dataset['bg'].detach().cpu(), nrow=5)
        self.writer.add_image(cat+'/BG Image', grid_image, global_step)
        grid_image = self.show( dataset['fg_bg'].detach().cpu(), nrow=5)
        self.writer.add_image(cat+'/FGBG Image', grid_image, global_step)

        grid_image = self.show( pred[1].detach().cpu(), nrow=5)
        self.writer.add_image(cat+'/Predicted mask', grid_image, global_step)
        grid_image = self.show( dataset['mask'].detach().cpu(), nrow=5)
        self.writer.add_image(cat+'/Groundtruth mask', grid_image, global_step)

        grid_image = self.show( pred[0].detach().cpu(), nrow=5)
        self.writer.add_image(cat+'/Predicted depth', grid_image, global_step)
        grid_image = self.show( dataset['depth'].detach().cpu(), nrow=5)
        self.writer.add_image(cat+'/Groundtruth depth', grid_image, global_step)
        del grid_image

    def LogEpoch(self, model, test_loader, epoch):
        model.eval()
        sequential = test_loader
        sample_batched = next(iter(sequential))
        bg_n    = sample_batched['bg'].to(device)
        fg_bg_n = sample_batched['fg_bg'].to(device)
        mask_n  = sample_batched['mask'].to(device)
        depth_n = sample_batched['depth'].to(device)
        if epoch == 0: self.writer.add_image('Test.1.BG', vutils.make_grid(sample_batched['bg'], nrow=6, normalize=True), epoch)
        if epoch == 0: self.writer.add_image('Test.2.FGBG', vutils.make_grid(sample_batched['fg_bg'], nrow=6, normalize=False), epoch)
        if epoch == 0: self.writer.add_image('Test.3.Mask', vutils.make_grid(sample_batched['mask'], nrow=6, normalize=False), epoch)
        if epoch == 0: self.writer.add_image('Test.4.Depth', vutils.make_grid(sample_batched['depth'], nrow=6, normalize=False), epoch)
        output =  model(bg_n, fg_bg_n)
        self.writer.add_image('Train.5.Pred_mask', vutils.make_grid(output[1].detach().cpu(), nrow=6, normalize=False), epoch)
        self.writer.add_image('Train.6.Pred_depth', vutils.make_grid(output[0].detach().cpu(), nrow=6, normalize=False), epoch)
        self.writer.add_image('Train.7.Diff_Mask', vutils.make_grid(torch.abs(output[1] - sample_batched['mask']).data, nrow=6, normalize=False), epoch)
        self.writer.add_image('Train.8.Diff_Depth', vutils.make_grid(torch.abs(output[0] - sample_batched['depth']).data, nrow=6, normalize=False), epoch)
        del sample_batched
        del output

    def show(self, tesonrs, *args, **kwargs):
        grid_tensor = torchvision.utils.make_grid( tesonrs[:15], *args, **kwargs)
        #grid_image  = grid_tensor.permute(1,2,0)
        return grid_tensor

    def close(self):
        self.writer.close()
    
    def save_checkpoint(self, model, val_acc_mask=None,  val_acc_depth=None):
        torch.save(model.state_dict(), f=self._last_cpt_path)
        if val_acc_mask is not None:
          if val_acc_mask > self._best_val_acc_mask:
              self._best_val_acc_mask = val_acc_mask
              torch.save(model.state_dict(), f=self._best_mask_cpt_path)
        if val_acc_depth is not None:
          if val_acc_depth > self._best_val_acc_depth:
              self._best_val_acc_depth = val_acc_depth
              torch.save(model.state_dict(), f=self._best_depth_cpt_path)

    def _create_dirs(self):
        """Create necessary directories"""
        for d in [
            self._tensorboard_dir,
            self._checkpoints_dir,
            # self._plots_dir,
            # self._val_disp_dir,
        ]:
            self._ensure_dir(d)

    def _ensure_dir(self, file: str) -> None:
        """
        Ensures that a given directory exists.
        Args:
            file: file
        """
        directory = os.path.dirname(file)
        if not os.path.exists(directory):
            os.makedirs(directory)