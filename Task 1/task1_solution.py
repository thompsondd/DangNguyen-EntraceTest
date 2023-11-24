from pathlib import Path
import cv2, math
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def imshow(img, showAxis = False, size=(20,10)):
    '''
    This function aims to plot img
    Args:
        img: An array of pixels
    '''
    plt.figure(figsize=size)
    if not showAxis: plt.axis('off')
    if len(img.shape) == 3: plt.imshow(img[:,:,::-1]) # Plot image with color
    else: plt.imshow(img, cmap='gray') # Plot image in gray color

def BBimgshow(imgFile,bound_box,return_img=False,show_img=True):
    img = cv2.imread(imgFile)
    '''
    This function aims to plot img with bounding box
    Args:
        imgFile: An image path
        bound_box: Info of bound box including x,y,w,h
        return_img: Return image with bouding box if true
        show_img: Plot image with bouding box if true
    '''
    x,y,w,h= bound_box
    cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,0), 4)
    if show_img: imshow(img)
    if return_img: return img

def rank_cnt(cnt):
    '''
    This function aims to score a contour.
    Ideas:
        - Because most of stamps in bills have a circle or ellipse shape and the ellipse is also a general form of circle, which means the ellipse will be circle if axis of ellipse have the same length, it is understandable to use ellipse shape as a contour bound.

        - I use three thresholds to check whether contour is valid or not.

        - The first threshold is ratio of area of ellipse bound and area of unstrict rectangle bound. As I observed, a valid stamp has this ratio smaller than 1, which means that the ellipse bound is inside unstrict rectangle bound.

        - The second threshold is ratio of area of ellipse bound and area of strict rectangle bound. As I observed, a valid stamp has this ratio smaller than 0.8, which means that the ellipse bound is inside strict rectangle bound.

        - The thrid threshold is ratio of two axis of ellipse bound. As I observed, a valid stamp has this ratio is between 0.47 and 1, which means that the shape of stamp could be an ellipse but it also should not have a dramatical difference between the length of axises.

        - The score is the area of the contour and the area of unstrict bounding box scaled by the ratio of two axis of ellipse bound.

    Args:
        imgFile: An image path
        bound_box: Info of bound box including x,y,w,h
        return_img: Return image with bouding box if true
        show_img: Plot image with bouding box if true
    '''
    if len(cnt)<5: return 0

    _,y_,w,h = cv2.boundingRect(cnt)
    area_boundingRect = w*h
    
    center, axes, angle = cv2.fitEllipse(cnt)
    semi_major_axis, semi_minor_axis = axes[0] / 2.0, axes[1] / 2.0
    area_ellipse = np.pi * semi_major_axis * semi_minor_axis
    ellipse_thres = (area_ellipse/area_boundingRect)

    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    area_box = cv2.contourArea(box)

    box_thres = area_ellipse/area_box if area_box!=0 else 2

    ellipse_axis_thres = semi_major_axis/semi_minor_axis
    ellipse_axis_thres = 1/ellipse_axis_thres if ellipse_axis_thres > 1 else ellipse_axis_thres

    if not(ellipse_thres < 1 and ellipse_axis_thres > 0.47 and box_thres<0.8): return 0

    return ellipse_axis_thres*area_boundingRect+cv2.contourArea(cnt)

def detect_model(imgFile, debug=False):
    '''
    This function aims to find the stamp.
    Ideas:
        - Step 1: Extract pixels as a mask has the value of red pixel account for 40%.
        - Step 2:
            2.1 Find contours in the mask
            2.2 Sort the contour based on the score from rank_cnt contour and and keep the best
            2.3 Create a new mask based on selected contour with the ellipse shape
        - Step 3
            3.1 Find the contours in the new mask
            3.2 Get the bounding box of exists contour

    Args:
        imgFile: An image path
        debug: Print the log if true
    '''

    img = cv2.imread(imgFile)

    #=====================================================================
    #Step 1: Extract pixels as a mask has the value of red pixel account for 40%.
    total = img.sum(axis=2)
    total[total==0]=1
    red_channel = img[:,:,2]

    red_mask = np.array((red_channel/total)>0.38, dtype=np.uint8)
    if debug: 
        print("Step 1: Detect color")
        imshow(red_mask)
    #=====================================================================
    #Step 2:
    
    #2.1 Find contours in the mask
    contours, _ = cv2.findContours(image=red_mask.copy(), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
   
    #2.2 Sort the contour based on the score from rank_cnt contour and and keep the best
    topN = 1
    sorted_contours = sorted(contours, key=rank_cnt, reverse=True)
    sorted_contours = sorted_contours[:topN]

    # 2.3 Create a new mask based on selected contour with the ellipse shape
    filteredCircle = np.zeros((img.shape[:2]), dtype =np.uint8)
    ellipse = cv2.fitEllipse(sorted_contours[0])
    cv2.ellipse(filteredCircle,ellipse,(250,100,100),-1)
    
    #=====================================================================
    # Step 3
    
    #3.1 Find the contours in the new mask
    filteredContours, _ = cv2.findContours(image=filteredCircle.copy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    
    #3.2 Get the bounding box of exists contour
    circleContours = []
    for _, contour in enumerate(filteredContours):
        circleContours.append(contour)
    
    bounds = circleContours[0].squeeze()
    max_x, max_y = bounds[:,0].max(), bounds[:,1].max()
    min_x, min_y = bounds[:,0].min(), bounds[:,1].min()
    bounds = circleContours[0].squeeze()
    w = max_x-min_x
    h = max_y-min_y

    return min_x,min_y,w,h

def stamp_detect(dst:str, src_dirs:list):
    for src in tqdm(src_dirs,desc="src_dirs"):
        type = os.path.split(src)
        save_dir = os.path.join(dst,type[-1])
        os.makedirs(save_dir,exist_ok=True)
        for root, dir, files in os.walk(src):
            for file in tqdm(files,desc="files",leave=True):
                try:
                    imgFile = os.path.join(root,file)
                    bound_box = detect_model(imgFile)
                    img = BBimgshow(imgFile,bound_box, return_img = True, show_img = False)
                    cv2.imwrite(os.path.join(save_dir,file), img)
                except Exception as e:
                    print(os.path.join(root,file))
                    raise e