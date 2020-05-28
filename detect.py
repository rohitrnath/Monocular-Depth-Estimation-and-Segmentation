import torch
import os
import glob
from pathlib import Path
from PIL import Image
import os
import shutil
from model.MDEASModel import *
# import tqdm
import cv2
import numpy as np
#from google.colab.patches import cv2_imshow

from torchvision.transforms import ToTensor
from torchvision import transforms

def getNoTransform():
    return transforms.Compose([
        transforms.Resize((224,224)),
        ToTensor()
    ])

img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']

class LoadImages:  # for inference
    def __init__(self, frames_path, bg_path):
        path = str(Path(frames_path))  # os-agnostic
        files = []
        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, '*.*')))
        elif os.path.isfile(path):
            files = [path]

        images = [x for x in files if os.path.splitext(x)[-1].lower() in img_formats]
        nI  = len(images)
        self.files = images
        self.nF = nI  # number of files
        self.bg = bg_path
        self.bg_transform    = getNoTransform()
        self.fg_bg_transform = getNoTransform()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.files)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        data = self.files[index]

        # Load data and get label
        fg_bg    = Image.open(data)
        bg      = Image.open(self.bg).convert('RGB')

        try:
          if self.bg_transform:
            bg = self.bg_transform(bg)
          if self.fg_bg_transform:
            fg_bg = self.fg_bg_transform(fg_bg)

        except:
          print('Error while transform:')
        
        sample = {'bg': bg, 'fg_bg': fg_bg, 'f_name': data}
        return sample

def fetchImageLoader( framesPath, bgPath, batchSize=60):
    dataset = LoadImages(framesPath, bgPath)
    input_loader = torch.utils.data.DataLoader(dataset, batch_size=batchSize,
                                              shuffle=False, num_workers=4)
    return input_loader



def runFrames(model, device, frameloader, out):
  model.eval()
  N = len(frameloader)
  with torch.no_grad():
    for i, sample_batch in enumerate(frameloader):
      bg    = sample_batch['bg'].to(device)
      fg_bg = sample_batch['fg_bg'].to(device)
      
      output = model(bg,fg_bg)

      for j in range(bg.size(0)):

        rescaled    =  fg_bg[j].detach().cpu().numpy()
        rescaled = rescaled - np.min(rescaled)
        rescaled = rescaled / np.max(rescaled)
        frame    = (rescaled*255).astype(np.uint8)
        frame = np.transpose(frame, (1, 2, 0))
        rescaled = output[0][j].detach().cpu().numpy()
        rescaled = rescaled - np.min(rescaled)
        rescaled = rescaled / np.max(rescaled)
        image1 = (rescaled*255).astype(np.uint8)
        image1 = np. transpose(image1, (1, 2, 0))
        image1 = cv2.applyColorMap(image1, cv2.COLORMAP_JET)

        rescaled = output[1][j].detach().cpu().numpy()
        rescaled = rescaled - np.min(rescaled)
        rescaled = rescaled / np.max(rescaled)
        image2 = (rescaled*255).astype(np.uint8)
        image2 = np.transpose(image2, (1, 2, 0))
        masked = cv2.bitwise_and(frame,frame,mask = image2)

        numpy_horizontal_concat1 = np.concatenate(( cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)), axis=1)
        numpy_horizontal_concat2 = np.concatenate(( cv2.cvtColor(image1, cv2.COLOR_RGB2BGR), cv2.cvtColor(masked, cv2.COLOR_RGB2BGR)), axis=1)
        numpy_vertical_concat    = np.concatenate((numpy_horizontal_concat1, numpy_horizontal_concat2), axis=0)

        # print(out+'/'+sample_batch['f_name'][i].split('/')[-1])
        # cv2_imshow(numpy_vertical_concat)
        cv2.imwrite(out+'/'+sample_batch['f_name'][j].split('/')[-1], numpy_vertical_concat)

      if i % (N//5) == 0:
        print(f'{i}/{N}')



def framesOnMDEAS(frame_path, bg_path, batchsize=50, weights ='/content/last-model.pth', out='out-out'):
  if os.path.exists(out):
      shutil.rmtree(out)  # delete output folder
  os.makedirs(out)  # make new output folder

  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  model = MDEASModel()
  print(device)
  print('Model created.')

  model = model.to(device)

  if device == 'cuda':
      print('cuda')
      model = torch.nn.DataParallel(model)
      cudnn.benchmark = True
  model.load_state_dict(torch.load(weights))

  frameloader = fetchImageLoader(frame_path, bg_path, batchsize)
  runFrames(model, device, frameloader, out)