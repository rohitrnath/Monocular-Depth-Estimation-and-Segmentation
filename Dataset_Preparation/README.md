# Create Dataset For Monocular Depth Estimation and Segmentation Model

Here we are creating custom dataset for monocular depth estimation and segmentation simultaneously

## Google Drive link to Dataset

### G-Drive Folder Structure

```bash
Assignment 15A
|-- Input
|   |-- bg (various size)
|   |   |-- bg1.jpg
|   |   |-- bg2.jpg
|   |   |-- .......
|   |   `-- bg100.jpg
|   |-- fg150
|   |   |-- fg_1.png
|   |   |-- fg_2.png
|   |   |-- .......
|   |   `-- fg_100.png
|   `-- fg_mask
|       |-- mask_fg_1.jpg
|       |-- mask_fg_2.jpg
|       |-- .......
|       `-- mask_fg_100.jpg
`-- Output
    |-- Dataset.zip
    `-- Dataset (Sample)
        |-- label_data.csv (bg, overlay, mask, depth)
        |-- bg (224*224)
        |   |-- bg1.jpg
        |   |-- bg2.jpg
        |   |-- .......
        |   `-- bg100.jpg
        |-- bg1
        |   |-- fg_1
        |   |   |-- overlay
        |   |   |   |-- 1.jpg
        |   |   |   |-- .......
        |   |   |   `-- 40.jpg
        |   |   |-- mask
        |   |   |   |-- 1.jpg
        |   |   |   |-- .......
        |   |   |   `-- 40.jpg
        |   |   `-- depth
        |   |       |-- 1.jpg
        |   |       |-- .......
        |   |       `-- 40.jpg
        |   |-- .......
        |   `-- fg_100
        |-- .......
        |-- bg100
        |   `-- .......
```



### Here is the link to custom dataset created :

https://drive.google.com/drive/folders/1fv71eSWdOED0plymX5kN9P_FuBcgdMCv?usp=sharing

* Dataset for the model is available in the Output folder(**Dataset.zip**)

* **Output/label_data.csv** file grouped all the image paths.

  Each row having background, Overlay, Mask, Depth images.

  This is useful to create training and testing dataset while modeling.

* Dataset.zip got unzipped to **Output/Dataset** . Some datas may be missing from this folder due to some google drive IO errors.

## Dataset Statistics

### Kinds of images

#### Inputs used to create Dataset

* Backgroud images of shape **224\*224\*3**  in **JPG** format.
* Foreground images with transparent background of  shape **M\*N\*4** in **PNG** format. (M and N can be anything , but **max(M,N) = 150**)
* Foreground mask images having same shape of foreground with single channel(**M\*N\*1**,  Grey-scale) and in **PNG** format.

#### Images in Dataset

* Overlayed images(fg_bg) of shape **224\*224\*3** in **JPG** format generated with quality=50%
* Mask images of shape **224\*224\*1**(Grey-scale) in **JPG** format generated with quality=50%
* Depth images of shape **224\*224\*1**(Grey-scale) in **JPG** format generated with quality=85%

### Total images of each kind

* 400000 fg_bg images

* 400K mask images

* 400K depth images

  **generated from**

  * 100 backgrounds

  * 100 foregrounds, plus their flips(2*100 = 200)

  * 20 random placement on each background.

    Total = 100\*200\*20 = 400000

  

### Total size of the dataset

* Dataset.zip of size **4.62 GB**.

  Which includes :-

  * label_data.csv of **47.9 MB**.
  * **120100 images** (100 backgrounds + 400K fg_bg + 400K masks + 400K depth)

### Mead and STD values

#### For fg_bg images

* Mean =    [149.17579724 143.51813416 136.34473418]
* STD    =    [10.9817084  10.54722837  9.7497292 ]

#### For mask images

* Mean =    19.611268351153935
* STD    =    21.938615940996442

#### For depth images

* Mean =    103.79459450852995
* STD    =    7.628609954872591

## Samples of Dataset

![all types](https://github.com/rohitrnath/Monocular-Depth-Estimation-and-Segmentation/tree/master/Dataset_Preparation/images_used/all_images.png)

## Steps To Create Custom Dataset

###Create Foreground Images with Transparent Background

####Method 1 - Using GIMP Tool

Followed this tutorial : https://www.youtube.com/watch?v=tKyRCS1qKTY

1. Load image to GIMP tool.

2. On the channels tab, check the image having alpha channel or not.

   If there is no alpha channel,

   * Select **Layer &#8594; Transparency &#8594; Add Alpha Channel**

    Alpha channel has been added to the image.

3. Using the **fuzzy select** tool or **region select** tool select the entire foreground that you want to keep.

4. Goto **Select &#8594; Invert** and press **delete** key.

5. Or we can select the background region that we want to remove and press **delete** key.

6. Crop the image keep the foreground image at center like a bounding box.

7. Resized the image so that the large side(max(height,width)) to **150px**.

8. Save image by select File  &#8594; Export As  &#8594; enter file name &#8594;  Export.

#### Method 2 - Using online background remove tools

1. Use online background tools such https://www.remove.bg to remove background.
2. For some images the automatic background removal may not seems to be perfect. So did some manual editing using the edit tab.
3. We can download the image by pressing download button. By default the image got 4 channels.
4. Loaded the background removed image to GIMP( for croping and to create the mask)
5. continued same steps 6,7,8 as mentioned above in method 1.

#### ![Sample Image](https://github.com/rohitrnath/Monocular-Depth-Estimation-and-Segmentation/tree/master/Dataset_Preparation/images_used/fg_imgs_1.png)

###Masks Created For Foregrounds

I used GIMP tool to create the mask. The steps are metioning below :

1. Load the  foreground image which having transparent background,resized and croped into GIMP tool.
2. The image should have 4 channels, as becuase we only creates this in the FG createion phase.
3. Using the **fuzzy select** tool click on the transparent region, that select entre background region excluding background.
4. Choose **black** color and paint black to the selected background region with **fill tool**.
5. Goto **Select &#8594; Invert** and fill the **foreground region with white**.
6. Save image by select File  &#8594; Export As  &#8594; enter file name &#8594;  Export.

___Observed some gradiant shift from black to white in the borders of the foreground mask___

So loaded the images to colab, made all non zero values to 255, and saved it as a grayscale image.

![Sample Image](https://github.com/rohitrnath/Monocular-Depth-Estimation-and-Segmentation/tree/master/Dataset_Preparation/images_used/fg_mask_imgs_1.png)

###Overlay Foreground Over Background



1. All background images are resized to (224*224) and saved to dataset as jpg with quality = 60%

2. For each foreground-background pare, we need 20 variants, including flipped foreground that become 40 variants.

3. The x and y position to overlay the foreground over background took randomly.

4. The foreground image resize to 4 variants. Five images each per variant.
   The 4 variants of foreground used are [actual ration, 0.8\*acual ratio, 0.6\*actual ration, 0.4\*actual ratio]

5. Steps to overlay foreground over background:

   * Took a copy of background
   * Resize the foreground to either of this [actual ration, 0.8\*acual ratio, 0.6\*actual ration, 0.4\*actual ratio]
   * Generate a random x,y position within image boundary and overlayed foreground on top of background.
   * Continue above step for 5 times to generate 5 overlayed images of same size. 

6. For creating the mask, first created a black rectangle with the same shape of background(224*224).
   Then resized the mask to the same ration that the foreground resized and overlayed on top of black rectangle in the same x,y position(randomly generated) that used for foreground overlay.

7. After generating one overlay-mask pair, save that to dataset with quality of 50%

8. **Overall process tooks around 20.8 minutes** to genarate all overlay and depth images

   

Wrote a python script to Overlay foreground over background and to create the respective mask.

```python
###PSEUDO CODE###

iterate through background image:
  resize bg_image to 224,224 and save to dataset.
  iterate through foreground and mask image:
    ratio=1
    for iteration of 2: #first iter with normat fg, other with flipped fg. Total=2*4*5=40img
      for iteration of 4: #to genarate 4 variants size of foregrounds
        resize fg_image to actual_size*ratio
        resize mask to actual_size*ratio
        ratio = ratio - 0.2
        for iteration of 5: #to generate 5*4=20 images with this fg-bg combination
          generate random x,y position, x,y within 0 to (bg_size - fg_size).

          overlay = paste resized fg_image on top of copy of background at x,y position.

          genarate black rectangle with shape of background image.
          mask = paste resized mask on top of black rectangle at x,y position.

          save both mask and overlay to dataset in jpg format with quality of 50%

          Update the label_data csv file with path to mask, overlay and depth which will 							create later
        
    	fg_image = flip fg_image horizontally(left to right) #flipping fg for next iteration
```



****

**Mask for fg_bg***

![mask](https://github.com/rohitrnath/Monocular-Depth-Estimation-and-Segmentation/tree/master/Dataset_Preparation/images_used/overlay_mask_imgs_1.png)

***Overlay foreground over background***

![Overlay](https://github.com/rohitrnath/Monocular-Depth-Estimation-and-Segmentation/tree/master/Dataset_Preparation/images_used/overlay_imgs_1.png)

### Create Depth Images

We used dense depth model to generate the depth images.

Made some changes in the existing script to run for our purpose.

The modified one is available here &#8594; https://github.com/rohitrnath/DenseDepth
Changes:

1. **tf.image.resize_image()** api got expired, so that replace with **tf.image.resize()** api in layers.py.
2. To support image of any input while testing, added a **resize to (640, 480)** call whiling loading the image.
3. The depth representation is changed from plasma to Grey for better visualisation.
4. New script file EVADepth.py added to generate all depth images while input the label_data csv file.
5. Inside EVADepth.denseDepthModel function, we saved the **output as a single channel grey scale image** in JPG format with quality of 85%.
6. With the **batch size of 60**, it took around **6.2 hours to generate 400K depth images**. All the input/output path it took from the csv feed.

Link to EVADepth.py - https://github.com/rohitrnath/DenseDepth/blob/master/EVADepth.py

####![mask](https://github.com/rohitrnath/Monocular-Depth-Estimation-and-Segmentation/tree/master/Dataset_Preparation/images_used/depth_imgs_1.png)

### ipython notebooks used

* **Notebook used to generate Overlayed image(fg_bg) and Mask**
  https://github.com/rohitrnath/Monocular-Depth-Estimation-and-Segmentation/tree/master/Dataset_Preparation//Assignment15A_Overlay.ipynb
* **Notebook used to generate Depth images**
  https://github.com/rohitrnath/Monocular-Depth-Estimation-and-Segmentation/tree/master/Dataset_Preparation/Assignment15A_Depth.ipynb
* **Notebook used to calculate statistics for this dataset**
  https://github.com/rohitrnath/Monocular-Depth-Estimation-and-Segmentation/tree/master/Dataset_Preparation/Assignment15A_Mean.ipynb
* **Notebook used to display sample images from dataset**
  https://github.com/rohitrnath/Monocular-Depth-Estimation-and-Segmentation/tree/master/Dataset_Preparation/Assignment15A_Display.ipynb







