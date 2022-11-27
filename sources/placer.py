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

max_intensity = 255
red_channel = 0
min_intensity_of_white_sheet = 180

step = 10

def mask_placer(rect, msk, area, max_x, max_y, dx, dy):
    for x in range(0, max_x, step):
        for y in range(0, max_y, step):
            if np.sum(cv2.bitwise_xor(msk, rect[x: x + dx, y : y + dy])) == area:
                rect[x: x + dx, y : y + dy] = msk
                return True
    return False
def placer(rect, masks, areas):
    rect = rect.astype(int)
    for msk, area in zip(masks, areas):
        msk = (~msk).astype(int)
        
        h, w = rect.shape
        dx, dy = msk.shape
        
        max_x = h - dx
        max_y = w - dy
        
        if mask_placer(rect, msk, area, max_x, max_y, dx, dy) is False:
            return False
    return True



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
    return s[0][0] > min_intensity_of_white_sheet

def find_poly(image):
    img = image.copy()

    inv_gray = ~cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(inv_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    
    if len(contours) != 0:
        contours = list(contours)
        contours.sort(key=cv2.contourArea, reverse=True)
        contour = contours[0]
        

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.004 * perimeter, True)
        
        
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
        
        cv2.drawContours(rect, [approx], -1, (1, 1, 1), -1)
        rect = rect[y:y+h, x:x+w]
    return img, box, rect


def filter_items(img, box):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU )

    inv_morph_th = ~binary_opening(th, footprint=np.ones((20, 20)))

    
    x,y,w,h = box

    inv_morph_th[y:y+h, x:x+w] = 0
    return inv_morph_th

def get_masks(wf):
    masks = []
    areas = []
    
    labels = label(wf)
    
    for i, region in enumerate(regionprops(labels)):
        if region.area >= min_area:
            mask = (labels == i + 1)
            x,y,xx,yy = region.bbox
            masks.append(mask[x:xx, y:yy])
            areas.append(region.area)
    return masks, areas
            