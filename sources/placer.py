import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

from imageio import imread, imsave
from skimage.morphology import binary_opening
from skimage.measure import regionprops
from skimage.measure import label
from scipy.ndimage import binary_fill_holes

min_area = 10000
min_area_figure = 1000

def get_images(path):
    images = []
    
    for file in os.listdir(path):
        if file.endswith(".jpg"):
            image = imread(os.path.join(path, file))
            images.append(image)

    return images

def calc_peak(rect):
    hist = cv2.calcHist([rect], [0], None, [256], [0, 256])
    hist = [val[0] for val in hist]
    indices = list(range(0, 256))
    s = [(x,y) for y,x in sorted(zip(hist,indices), reverse=True)]
    return s[0][0] > 180

def find_poly(image):
    img = image.copy()

    inv_gray = ~cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(inv_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    
    if len(contours) != 0:
        contours = list(contours)
        contours.sort(key=cv2.contourArea, reverse=True)
        contour = contours[0]
        #contour = max(contours, key = cv2.contourArea)

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.004 * perimeter, True)
        #cv2.drawContours(img, [approx], -1, (255, 255, 255), 30)
        
        box = cv2.boundingRect(approx)
        x,y,w,h = box
        isWhiteInside = calc_peak(img[y:y+h, x:x+w])
        
        k = 0
        while isWhiteInside == False:
            k = k + 1
            contour = contours[k]
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.004 * perimeter, True)
            box = cv2.boundingRect(approx)
            x,y,w,h = box
            isWhiteInside = calc_peak(img[y:y+h, x:x+w])
            
        if(cv2.contourArea(contour) < min_area_figure):
            return None, None, None
        rect = np.zeros_like(inv_gray)
        
        cv2.drawContours(rect, [approx], -1, (255, 255, 255), -1)
        rect = rect[y:y+h, x:x+w]
    return img, box, rect


def without_figures(img, box):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU )

    inv_morph_th = ~binary_opening(th, footprint=np.ones((20, 20)))

    
    x,y,w,h = box

    inv_morph_th[y:y+h, x:x+w] = 0
    return inv_morph_th

def get_masks(wf):
    masks = []
    labels = label(wf)
    
    for i, region in enumerate(regionprops(labels)):
        if region.area >= min_area:
            mask = (labels == i + 1)
            x,y,xx,yy = region.bbox
            masks.append(mask[x:xx, y:yy])
    return masks
            