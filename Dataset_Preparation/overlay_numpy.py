import cv2
import numpy as np

def overlay_transparent(mainimage, overlay, x, y):

    background  = mainimage.copy()

    background_width = background.shape[1]
    background_height = background.shape[0]

    maskBG = np.zeros([background_width, background_height, background.shape[2]], dtype = np.uint8)

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        # w = background_width - x
        x = background_width - w
        overlay = overlay[:, :w]

    if y + h > background_height:
        # h = background_height - y
        y = background_height - h
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
            ],
            axis = 2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    maskBG[y:y+h, x:x+w] = ((1.0 - mask) * maskBG[y:y+h, x:x+w] + mask * 255)

    return background, maskBG


from random import randint
import cv2
import numpy as np

path_FG_BG = 'Output/FG_BG'
path_mask_FG_BG = 'Output/mask_FG_BG'

quadrantsX  = [ 100, 0, 100, 0]
quadrantsY  = [ 100, 100, 0, 0]

def makeCombinations( backgound, foreground, opDir, imageNo):

  fg_img_tiny = cv2.resize(foreground, (foreground.shape[0], foreground.shape[1]), interpolation = cv2.INTER_AREA)
  for i in range(0,2):
    x = randint(0, 199 - (fg_img_tiny.shape[0]%199))
    y = randint(0, 199 - (fg_img_tiny.shape[1]%199))
    fg_bg, masked = overlay_transparent(backgound, fg_img_tiny, x, y)
    
    imageNo = imageNo + 1
    mask_grey = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(opDir+'/overlay/'+str(imageNo)+".jpg", fg_bg)
    cv2.imwrite(opDir+'/mask/'+str(imageNo)+".jpg", mask_grey)
    
    # cv2_imshow(fg_bg)
    # cv2_imshow(masked)


  fg_img_tiny = cv2.resize(foreground, (int(0.8*foreground.shape[0]), int(0.8*foreground.shape[1])), interpolation = cv2.INTER_AREA)
  for i in range(0,4):
    x = randint( quadrantsX[i%4], quadrantsX[i%4] + 99 - (fg_img_tiny.shape[0]%99) )
    y = randint( quadrantsY[i%4], quadrantsY[i%4] + 99 - (fg_img_tiny.shape[1]%99))
    fg_bg, masked = overlay_transparent(backgound, fg_img_tiny, x, y)
    
    imageNo = imageNo + 1
    mask_grey = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(opDir+'/overlay/'+str(imageNo)+".jpg", fg_bg)
    cv2.imwrite(opDir+'/mask/'+str(imageNo)+".jpg", mask_grey)

    # cv2_imshow(fg_bg)
    # cv2_imshow(masked)

  fg_img_tiny = cv2.resize(foreground, (int(0.6*foreground.shape[0]), int(0.6*foreground.shape[1])), interpolation = cv2.INTER_AREA)
  for i in range(0,5):
    x = randint( quadrantsX[i%4], quadrantsX[i%4] + 99 - (fg_img_tiny.shape[0]%99) )
    y = randint( quadrantsY[i%4], quadrantsY[i%4] + 99 - (fg_img_tiny.shape[1]%99) )
    fg_bg, masked = overlay_transparent(backgound, fg_img_tiny, x, y)
    
    imageNo = imageNo + 1
    mask_grey = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(opDir+'/overlay/'+str(imageNo)+".jpg", fg_bg)
    cv2.imwrite(opDir+'/mask/'+str(imageNo)+".jpg", mask_grey)

    # cv2_imshow(fg_bg)
    # cv2_imshow(masked)

  x = 100 - (int(fg_img_tiny.shape[0]/2))
  y = 100 - (int(fg_img_tiny.shape[1]/2))
  fg_bg, masked = overlay_transparent(backgound, fg_img_tiny, x, y)
  
  imageNo = imageNo + 1
  mask_grey = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
  cv2.imwrite(opDir+'/overlay/'+str(imageNo)+".jpg", fg_bg)
  cv2.imwrite(opDir+'/mask/'+str(imageNo)+".jpg", mask_grey)

  # cv2_imshow(fg_bg)
  # cv2_imshow(masked)

  fg_img_tiny = cv2.resize(foreground, (int(0.5*foreground.shape[0]), int(0.5*foreground.shape[1])), interpolation = cv2.INTER_AREA)
  for i in range(0,5):
    x = randint( quadrantsX[i%4], quadrantsX[i%4] + 99 - (fg_img_tiny.shape[0]%99))
    y = randint( quadrantsY[i%4], quadrantsY[i%4] + 99 - (fg_img_tiny.shape[1]%99))
    fg_bg, masked = overlay_transparent(backgound, fg_img_tiny, x, y)
    
    imageNo = imageNo + 1
    mask_grey = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(opDir+'/overlay/'+str(imageNo)+".jpg", fg_bg)
    cv2.imwrite(opDir+'/mask/'+str(imageNo)+".jpg", mask_grey)

    # cv2_imshow(fg_bg)
    # cv2_imshow(masked)


  x = 100 - (int(fg_img_tiny.shape[0]/2))
  y = 100 - (int(fg_img_tiny.shape[1]/2))
  fg_bg, masked = overlay_transparent(backgound, fg_img_tiny, x, y)
  
  imageNo = imageNo + 1
  mask_grey = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
  cv2.imwrite(opDir+'/overlay/'+str(imageNo)+".jpg", fg_bg)
  cv2.imwrite(opDir+'/mask/'+str(imageNo)+".jpg", mask_grey)

  # cv2_imshow(fg_bg)
  # cv2_imshow(masked)

  fg_img_tiny = cv2.resize(foreground, (int(0.3*foreground.shape[0]), int(0.3*foreground.shape[1])), interpolation = cv2.INTER_AREA)
  for i in range(0,2):
    x = randint( quadrantsX[i%4], quadrantsX[i%4] + 99 - (fg_img_tiny.shape[0]%99))
    y = randint( quadrantsY[i%4], quadrantsY[i%4] + 99 - (fg_img_tiny.shape[1]%99))
    fg_bg, masked = overlay_transparent(backgound, fg_img_tiny, x, y)
    
    imageNo = imageNo + 1
    mask_grey = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(opDir+'/overlay/'+str(imageNo)+".jpg", fg_bg)
    cv2.imwrite(opDir+'/mask/'+str(imageNo)+".jpg", mask_grey)

    # cv2_imshow(fg_bg)
    # cv2_imshow(masked)











from os import listdir
from DenseDepth.EVADepth import denseDepthModel
# from google.colab.patches import cv2_imshow
# import cv2
from matplotlib.pyplot import imshow
import numpy as np
from PIL import Image

%matplotlib inline
import os, errno

path_BG = 'bg'
path_FG = 'fg150'
bg_imageListDir = listdir(path_BG)
fg_imageListDir = listdir(path_FG)
print(len(bg_imageListDir))
for bg_image in bg_imageListDir:
  print(bg_image)
  outputDir = 'Dataset/'
  outputDir = outputDir + bg_image[:-4]
  bg_img = Image.open(path_BG+'/'+bg_image).resize((224,224), Image.ANTIALIAS)
  print(bg_img.size)
  # cv2_imshow(bg_img)
  for fg_image in fg_imageListDir:
    fg_img = Image.open(path_FG+'/'+fg_image).resize((224,224), Image.ANTIALIAS)
    break
  break
    # outputDir1 = outputDir + '/' + fg_image[:-4]
    # try:
    #     os.makedirs(outputDir1 + "/overlay")
    #     os.makedirs(outputDir1 + "/mask")
    #     os.makedirs(outputDir1 + "/depth")
    # except FileExistsError:
    #     pass

    # makeCombinations( bg_img, fg_img, outputDir1, 0)
    # #Flip foreground
    # fg_img_flip = cv2.flip(fg_img, 1)
    # makeCombinations( bg_img, fg_img_flip, outputDir1, 20)
    # denseDepthModel( model, outputDir1 + '/overlay/*.jpg', outputDir1 + '/depth/')