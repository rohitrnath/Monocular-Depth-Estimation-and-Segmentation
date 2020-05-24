import argparse
import os

import time
import datetime

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.utils as utils

from dataloader.dataloader import GetTrainTestData
from model.MDEASModel import MDEASModel
from Logger.SummaryTracker import SummaryTracker
from criterion.SSIM import SSIM
from criterion.mIoU import mIoU
from LR_Scheduler.cyclicLR import cyclicLR
from utils.utils import AverageMeter

import torchvision.utils as vutils
from torchsummary import summary

class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Tensorboard Summary
        self.summary = SummaryTracker(args.logdir)
        
        # Define Dataloader
        self.train_loader, self.test_loader = GetTrainTestData( arg.dataset, arg.ratio, trainBS=args.batch_size, testBS=args.test_batch_size, DEBUG=args.debug)

        # Define network
        self.model = MDEASModel()

        # Define Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(),  args.lr, betas=(0.5,0.999))

        # Define Criterion
        self.criterion_ssim = SSIM()
        self.criterion_mse  = nn.MSELoss()
        self.criterion_bce  = nn.BCEWithLogitsLoss()
        self.criterion_l1   = nn.L1Loss()
        
        # Define lr scheduler
        self.scheduler = cyclicLR(self.optimizer, lr_min=args.lr_min, lr_max=args.lr_max, batch_size=args.batch_size, startEpoch=args.start_epoch, epochs=args.edge_len, MaxNumCycles=arg.cycles, constEpochs= arg.warmup, constLR= args.lr )

        # Using cuda
        self.device = args.device
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model)
            self.model = self.model.to(device)

        # Resuming checkpoint
        self.best_val_acc_mask = [ 0.0, 0.0]
        self.best_val_acc_depth = [ 0.0, 0.0]
        if args.resume is not None:
            self.model.load_state_dict(torch.load(args.resume))

        self.train_start_time = time.time()


    def training(self, epoch):
        # Init training loss
        batch_time = AverageMeter()
        losses     = AverageMeter()
        losses_depth = AverageMeter()
        losses_mask  = AverageMeter()
        losses_l1depth = AverageMeter()
        losses_l1mask = AverageMeter()
        train_Acc_Mask     = AverageMeter()
        train_Acc_Depth     = AverageMeter()
        
        epoch_time = time.time()

        N = len(m_train_loader)
        self.model.train()
        for i, sample_batched in enumerate(m_train_loader):
          lr = self.scheduler.step(epoch,i)
          #Prepare sample and target
          bg_n    = sample_batched['bg'].to(self.device)
          fg_bg_n = sample_batched['fg_bg'].to(self.device)
          mask_n  = sample_batched['mask'].to(self.device)
          depth_n = sample_batched['depth'].to(self.device)

          # depth_n = DepthNorm( depth_n )
          # mask_n = DepthNorm( mask_n )

          # One optimization iteration
          self.optimizer.zero_grad()

          # Predict
          output = self.model( bg_n, fg_bg_n)

          # Compute the loss
          l_mask_acc  = self.criterion_ssim(output[1], mask_n)
          l_mask      = 1-l_mask_acc
          l_mask2     = self.criterion_mse(output[1], mask_n)
          l1_mask     = criterion_l1(output[1], mask_n)
          l_depth_acc = self.criterion_ssim(output[0], depth_n)
          l_depth     = 1-l_depth_acc
          l_depth2    = self.criterion_mse(output[0], depth_n)
          l1_depth    = criterion_lq(output[0], depth_n)

          #loss =  (1.0 * l_depth) + (0.00001 * l1_mask)+ (0.00001 * l1_depth) + (.5*l_mask2) #+(1.5 * l_depth.item()) + (0.1*l_depth2) 
          #loss =  (2.0 * l_depth) + (1. * l_mask2) + (0.00001 * l1_depth )+ (0.00001 * l1_mask)
          loss =  (2.0 * l_depth) +(.4 * l_mask) + (.1 * l_depth2) + (1. * l_mask2) + (0.000001 * l1_depth )+ (0.00001 * l1_mask)
          # Update step
          loss.backward()
          self.optimizer.step()
        
          losses.update(loss.data.item(), bg_n.size(0))
          losses_depth.update(l_depth.data.item(), bg_n.size(0))
          losses_mask.update(l_mask.data.item(), bg_n.size(0))
          losses_l1depth.update(l1_depth.data.item(), bg_n.size(0))
          losses_l1mask.update(l1_mask.data.item(), bg_n.size(0))

          #Measure Accuracy
          # acc_depth = mIoU( output[0], depth_n)
          # acc_mask  = mIoU( output[1], mask_n)
          train_Acc_Mask.update(l_mask_acc.data.item(), fg_bg_n.size(0))
          train_Acc_Depth.update(l_depth_acc.data.item(), fg_bg_n.size(0))

          

          # # Measure elapsed time
          # batch_time.update(time.time() - end)
          # end = time.time()
          # eta = str(datetime.timedelta(seconds=int(batch_time.val*(N - i))))
          # pbar1.set_description(desc = f'[{epoch}] loss={loss.item()} mask={l_mask.item()} depth={l_depth.item()}')
          if i% (N//20) == 0:
          # if i % 50 == 0:
            print(f'[{epoch}][{i}/{N}] loss={loss.item()} mask_ssim={l_mask.item()} depth_ssim={l_depth.item()} mask_l1={l1_mask.item()} depth_l1={l1_depth.item()} mask_mse={l_mask2.item()} depth_mse={l_depth2.item()} Acc-mask={l_mask_acc.data.item()}  Acc-depth={l_depth_acc.data.item()}  Epoch Time={time_delta_now(epoch_time)}')
          
          # Log progress
          if i % 50 == 0:
            global_step = epoch*N+i
            # Write to summary
            self.summary.addToSummary('Global/Loss', losses.val, global_step)
            self.summary.addToSummary('Global/Mask_Loss_ssim', losses_mask.val, global_step)
            self.summary.addToSummary('Global/Depth_Loss_ssim', losses_depth.val, global_step)
            self.summary.addToSummary('Global/Mask_Loss_l1', losses_l1depth.val, global_step)
            self.summary.addToSummary('Global/Depth_Loss_l1', losses_l1mask.val, global_step)

            self.summary.addToSummary('Global/Mask_Acc', train_Acc_Mask.val, global_step)
            self.summary.addToSummary('Global/Depth_Acc', train_Acc_Depth.val, global_step)

            if i % 500:
              self.summary.visualize_image("Global",sample_batched, output, global_step)
              # if i% 500:
              self.summary.save_checkpoint(model)
 
        #################
        # Track results #
        #################
        self.summary.visualize_image("Training",sample_batched, output, epoch)
        #Log Train Epoch
        self.summary.addToSummary('Loss/train', losses.avg, epoch)
        self.summary.addToSummary('Mask_Acc/train',   train_Acc_Mask.avg, epoch)
        self.summary.addToSummary('Depth_Acc/train',  train_Acc_Depth.avg, epoch)
        self.summary.addToSummary('Mask_Loss/train',  losses_mask.avg, epoch)
        self.summary.addToSummary('Depth_Loss/train', losses_depth.avg, epoch)
        self.summary.addToSummary('l1_Mask_Loss/train', losses_l1depth.avg, epoch)
        self.summary.addToSummary('l1_Depth_Loss/train', losses_l1mask.avg, epoch)
        #print in console
        print('Epoch: [{0}]\t'
              'Epoch Time={epochTime}\t'
              'Time Drift={timeDrift}\t'
              'Loss {losses.avg:.4f}\t'
              'Mask Loss={losses_mask.avg:.4f}  Depth Loss={losses_depth.avg:.4f}\t'
              'Mask Acc={train_Acc_Mask.avg:.4f}  Depth Acc={train_Acc_Depth.avg:.4f}\t'
              .format(epoch, epochTime=time_delta_now(epoch_time), timeDrift=time_delta_now(self.train_start_time),
                      losses=losses, losses_mask=losses_mask, train_Acc_Mask=train_Acc_Mask, losses_depth=losses_depth, 
                      train_Acc_Depth=train_Acc_Depth))


    def validation(self, epoch, eval_limit):
        # Init validation loss
        val_losses     = AverageMeter()
        val_losses_depth = AverageMeter()
        val_losses_mask  = AverageMeter()
        val_losses_l1depth = AverageMeter()
        val_losses_l1mask = AverageMeter()
        val_Acc_Depth = AverageMeter()
        val_Acc_Mask = AverageMeter()
        val_mIoU_Depth = AverageMeter()
        val_mIoU_Mask = AverageMeter()
        #Validations starting    
        val_start_time = time.time()
        self.model.eval()
        with torch.no_grad():
          N = len(self.m_test_loader)
          # pbar2 = tqdm(m_test_loader)
          for i, sample_batch in enumerate(self.m_test_loader):
            bg_n    = sample_batch['bg'].to(self.device)
            fg_bg_n = sample_batch['fg_bg'].to(self.device)
            mask_n  = sample_batch['mask'].to(self.device)
            depth_n = sample_batch['depth'].to(self.device)

            # depth_n = DepthNorm( depth_n )
            # mask_n = DepthNorm( mask_n )

            output = self.model( bg_n, fg_bg_n)

            # Compute the loss
            l_mask_acc  = self.criterion_ssim(output[1], mask_n)
            l_mask      = 1-l_mask_acc
            l_mask2     = self.criterion_mse(output[1], mask_n)
            l1_mask     = criterion_l1(output[1], mask_n)
            l_depth_acc = self.criterion_ssim(output[0], depth_n)
            l_depth     = 1-l_depth_acc
            l_depth2    = self.criterion_mse(output[0], depth_n)
            l1_depth    = criterion_lq(output[0], depth_n)

            #loss =  (1.0 * l_depth) + (0.00001 * l1_mask)+ (0.00001 * l1_depth) + (.5*l_mask2) #+(1.5 * l_depth.item()) + (0.1*l_depth2) 
            #loss =  (2.0 * l_depth) + (1. * l_mask2) + (0.00001 * l1_depth )+ (0.00001 * l1_mask)
            loss =  (2.0 * l_depth) +(.4 * l_mask) + (.1 * l_depth2) + (1. * l_mask2) + (0.000001 * l1_depth )+ (0.00001 * l1_mask)


            # pbar2.set_description(desc = f'[{epoch}] loss={loss.item()} mask={l_mask.item()} depth={l_depth.item()}')

            val_losses.update(loss.data.item(), bg_n.size(0))
            val_losses_depth.update(l_depth.data.item(), bg_n.size(0))
            val_losses_mask.update(l_mask.data.item(), bg_n.size(0))
            val_losses_l1depth.update(l1_depth.data.item(), bg_n.size(0))
            val_losses_l1mask.update(l1_mask.data.item(), bg_n.size(0))

            #Measure mean IoU
            mIoU_mask  = mIoU(output[1], mask_n)
            mIoU_Depth = mIoU(output[0], depth_n)
            val_mIoU_Mask.update(mIoU_mask, fg_bg_n.size(0))
            val_mIoU_Depth.update(mIoU_Depth, fg_bg_n.size(0))

            #Measure Accuracy
            acc_depth = l_depth_acc.item()
            acc_mask  = l_mask_acc.item()
            val_Acc_Mask.update(acc_mask, fg_bg_n.size(0))
            val_Acc_Depth.update(acc_depth, fg_bg_n.size(0))

            if -1 != eval_limit:
              if i >= eval_limit:
                break

            if i% (N//5) == 0:
              print(f'[{epoch}][{i}/{N}]  Acc-mask={acc_mask}  Acc-depth={acc_depth} mIoU-mask={mIoU_mask} mIoU-mask={mIoU_Depth} Epoch Time={time_delta_now(val_start_time)}')
  
        show(sample_batch['depth'].cpu(), nrow=5)
        show(output[0].detach().cpu(), nrow=5)
        show(sample_batch['fg_bg'].cpu(), nrow=5)
        show(sample_batch['mask'].cpu(), nrow=5)
        show(output[1].detach().cpu(), nrow=5)
        print('Epoch: [{0}][{1}/{2}]\t'
          'Valid Time={validTime}\t'
          'Time Drift={timeDrift}\t'
          'Mask IoU={val_IoU_Mask:.4f}  Depth IoU={val_IoU_Depth:.4f}\t'
          'Mask Acc={val_Acc_Mask:.4f}  Depth Acc={val_Acc_Depth:.4f}\t'
          'Loss {losses.avg:.4f}\t'
          'Mask Loss={losses_mask.avg:.4f}  Depth Loss={losses_depth.avg:.4f}\t\n\n'
          .format(epoch, i, N, validTime=time_delta_now(val_start_time), timeDrift=time_delta_now(train_start_time),
                  losses=val_losses, losses_mask=val_losses_mask, val_Acc_Mask=val_Acc_Mask.avg, losses_depth=val_losses_depth, 
                  val_Acc_Depth=val_Acc_Depth.avg, val_IoU_Mask=val_mIoU_Mask.avg, val_IoU_Depth= val_mIoU_Depth.avg))
        
        if eval_limit != -1:
          #################
          # Track results #
          #################
          self.summary.visualize_image("validation",sample_batch, output, epoch)
          #Log Validation Epoch
          self.summary.addToSummary('Loss/valid', val_losses.avg, epoch)
          self.summary.addToSummary('Mask_Acc/valid',   val_Acc_Mask.avg, epoch)
          self.summary.addToSummary('Depth_Acc/valid',  val_Acc_Depth.avg, epoch)
          self.summary.addToSummary('Mask_mIoU/valid',   val_mIoU_Mask.avg, epoch)
          self.summary.addToSummary('Depth_mIoU/valid',  val_mIoU_Depth.avg, epoch)
          self.summary.addToSummary('Mask_Loss/valid',  val_losses_mask.avg, epoch)
          self.summary.addToSummary('Depth_Loss/valid', val_losses_depth.avg, epoch)
          self.summary.addToSummary('l1_Mask_Loss/valid', val_losses_l1depth.avg, epoch)
          self.summary.addToSummary('l1_Depth_Loss/valid', val_losses_l1mask.avg, epoch)

          if(val_Acc_Mask.avg > self.best_val_acc_mask[0]):
            self.best_val_acc_mask = [val_Acc_Mask.avg,epoch]
          if(val_Acc_Depth.avg > self.best_val_acc_depth[0]):
            self.best_val_acc_depth = [val_Acc_Depth.avg,epoch]

          self.summary.save_checkpoint( self.model, val_Acc_Mask.avg, val_Acc_Depth.avg)


  print(f"\n\nFinished Training. Best Mask Acc: {best_val_acc_mask[0]} @ epoch {best_val_acc_mask[1]}")
  print(f"Finished Training. Best Depth Acc: {best_val_acc_depth[0]} @ epoch {best_val_acc_depth[1]}\n")

  self.summary.close()



def main():
    parser = argparse.ArgumentParser(description="PyTorch MDEAS model Training")

    parser.add_argument('--dataset', type=str, default='Dataset/label_data.csv',
                        help='path to dataInfo file')
    parser.add_argument('--debug', action='store_true', default=
                    False, help='debug with 1000K images')
    parser.add_argument('--logdir', type=str, default='/content',
                        help='path to Tensorboard Logger')
    parser.add_argument('--ratio', type=float, default=0.7,
                        help='Train:Test data ratio( default : 0.7)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=224,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=224,
                        help='crop image size')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=128,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=128,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: auto)')

    # scheduler params
    parser.add_argument('--lr_min', type=float, default=0.00001, metavar='minLR',
                        help='min LR for cyclic LR scheduler')
    parser.add_argument('--lr_max', type=float, default=0.001, metavar='maxLR',
                        help='max LR for cyclic LR scheduler')
    parser.add_argument('--cycles', type=int, default=3, metavar='C',
                        help='number of cycles for cyclicLR scheduler')
    parser.add_argument('--warmup', type=int, default=5, metavar='W',
                        help='warm-up epochs before cyclicLR scheduler')
    parser.add_argument('--edge', type=int, default=5, metavar='E',
                        help='Edge epochs for cyclicLR scheduler')


    # cuda, seed and logging
    parser.add_argument('--device', type=str, default='cuda',
                        help='use cpu/cuda')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    # evaluation option
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')
    parser.add_argument('--eval-limit', type=int, default=-1,
                        help='Evaluate only first n batches')

    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if use_cuda else "cpu")

    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)

    eval_limit = args.eval_limit
    
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    print(f"Training starts at {datetime.datetime.now()} ")

    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val 
            trainer.validation( epoch, eval_limit)

    trainer.summary.close()







if __name__ == "__main__":
   main()