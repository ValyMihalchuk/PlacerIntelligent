import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

from imageio import imread, imsave
from skimage.morphology import binary_opening
from skimage.measure import regionprops
from skimage.measure import label
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import rotate

from skimage.morphology import binary_closing

# The required parameters
min_area = 10000 # min area of items
min_area_figure = 1000 # min area of poly

max_intensity = 255 # max intensity
red_channel = 0 #index of red channel
min_intensity_of_white_sheet = 180 # min intensity of white sheet in red channel

step = 10 # step for x,y and angle in mask_placer function



def my_xor(mask, rect):
    result = mask.copy()
    for i in range(len(mask)):
        for j in range(len(mask[i])):
            #print(mask[i][j])
            #print(rect[i][j])
            if mask[i][j]==0:
                result[i][j]=int(bool(mask[i][j]) != bool(rect[i][j]))
            #print(result[i][j])   
    return result
# The mask_placer function - it receives a polygon ans a mask with areas as input. She applies the mask to the rect until it fits
def mask_placer(rect, msk, area):
    h, w = rect.shape
    msk = msk.astype(int)
    msk = msk ^ 1
    
    for angle in range(0, 180, step):
        rotated_mask = msk.astype(int)
        rotated_mask = rotate(rotated_mask, angle, reshape=True)
        
        
        dx, dy = rotated_mask.shape
        #plt.imshow(rotated_mask)
        #plt.show()
        max_x = h - dx # max value for x
        max_y = w - dy # max value for y
        
        for x in range(0, max_x, step):
            for y in range(0, max_y, step):

                # our polygon is filled with white and the subject is black - so if we sum result, we get masks area if mask fits
                if np.sum(cv2.bitwise_xor(rotated_mask ^ 1, rect[x: x + dx, y : y + dy])) == area:
                    rect[x: x + dx, y : y + dy] = rotated_mask ^ 1 #now in rect there are our mask

                    return True
    return False

#The mask_placer function - it receives a polygon and a masks with areas as input. She applies each mask to the rect
def placer(rect, masks, areas):
    rect = rect.astype(int)
    
    for msk, area in zip(masks, areas):
            if mask_placer(rect, ~msk, area) is False:
                return False, rect
    return True, rect


# Just get images
def get_images(path):
    images = []
    
    for file in os.listdir(path):
        if file.endswith(".jpg"):
            image = imread(os.path.join(path, file))
            images.append(image)

    return images

# Calc peak function - it finds the histogram extremum in the red channel, and returns true if that extremum corresponds to intensity of white paper.
def calc_peak(rect):
    hist = cv2.calcHist([rect], [0], None, [256], [0, 256])
    hist = [val[0] for val in hist]
    indices = list(range(0, 256))
    s = [(x,y) for y,x in sorted(zip(hist,indices), reverse=True)]
    return s[0][0] > min_intensity_of_white_sheet

# finds polygon in image and returns bounding box, image with only polygon
def find_poly(image):
    img = image.copy()

    # First find contours before that we apply binarization
    inv_gray = ~cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(inv_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    
    if len(contours) != 0:
        #Get max area contour
        contours = list(contours)
        contours.sort(key=cv2.contourArea, reverse=True)
        contour = contours[0]
        

        # Approximate by poly this
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.004 * perimeter, True)
        
        # Find bound box
        box = cv2.boundingRect(approx)
        x,y,w,h = box
        isWhiteInside = calc_peak(img[y:y+h, x:x+w]) #check is this paper or polygon
        
        #if not, take another contour
        k = 0
        while isWhiteInside == False:
            k = k + 1
            contour = contours[k]
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.004 * perimeter, True)
            box = cv2.boundingRect(approx)
            x,y,w,h = box
            isWhiteInside = calc_peak(img[y:y+h, x:x+w])
        # Don't have a polygon? Return nothing    
        if(cv2.contourArea(contour) < min_area_figure):
            return None, None
        
        # Now draw only polgon
        rect = np.zeros_like(inv_gray)
        cv2.drawContours(rect, [approx], -1, (1, 1, 1), -1)
        rect = rect[y:y+h, x:x+w]
    return box, rect


def filter_items(img, box):
    # Binarization
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU )

    # Morphology opening/closing for better results 
    inv_morph_th = ~binary_opening(th, footprint=np.ones((20, 20)))
    inv_morph_th = binary_closing(inv_morph_th, footprint=np.ones((30, 30)))
    x,y,w,h = box

    # Fill polygon black, no longer need
    inv_morph_th[y:y+h, x:x+w] = 0
    return inv_morph_th

def get_masks(wf):
    masks = []
    areas = []
    
    # labels - matrix filled with indices of connectivity components
    labels = label(wf)
    
    for i, region in enumerate(regionprops(labels)):
        if region.area >= min_area:
            mask = (labels == i + 1) #get mask with our index
            x,y,xx,yy = region.bbox
            masks.append(mask[x:xx, y:yy])
            areas.append(region.area)
    return masks, areas
            