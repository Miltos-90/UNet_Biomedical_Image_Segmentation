
from collections import defaultdict
import numpy as np
import cv2
import os
from PIL import Image

def readData(datadir, labeldir, H = 512, W = 512):
    img_mtrx  = np.empty(shape = (30, H, W), dtype = np.uint8)
    mask_mtrx = np.empty(shape = (30, H, W))

    # Loop over all image / mask pairs
    for i, (img, lab) in enumerate(zip(os.listdir(datadir), os.listdir(labeldir))):
            
        # Load image
        image = np.array(Image.open(datadir + img))
        
        # Add channel dimension
        image = np.expand_dims(image, axis = 0)
        
        # Add to matrix
        img_mtrx[i, :, :] = image 
        
        # Load mask in grayscale (single channel)
        mask = cv2.imread(labeldir + lab, cv2.IMREAD_GRAYSCALE)
    
        # Binarise
        _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
            
        # Add to matrix
        mask_mtrx[i, :, :] = mask 

    return img_mtrx, mask_mtrx

def weight_map(mask, w0, sigma, background_class = 0):
    
    # Fix mask datatype (should be unsigned 8 bit)
    if mask.dtype != 'uint8': 
        mask = mask.astype('uint8')
    
    # Weight values to balance classs frequencies
    wc = _class_weights(mask)
    
    # Assign a different label to each connected region of the image
    _, regions = cv2.connectedComponents(mask)
    
    # Get total no. of connected regions in the image and sort them excluding background
    region_ids = sorted(np.unique(regions))
    region_ids = [region_id for region_id in region_ids if region_id != background_class]
        
    if len(region_ids) > 1: # More than one connected regions

        # Initialise distance matrix (dimensions: H x W x no.regions)
        distances = np.zeros((mask.shape[0], mask.shape[1], len(region_ids)))

        # For each region
        for i, region_id in enumerate(region_ids):

            # Mask all pixels belonging to a different region
            m = (regions != region_id).astype(np.uint8)# * 255
        
            # Compute Euclidean distance for all pixels belongind to a different region
            distances[:, :, i] = cv2.distanceTransform(m, distanceType = cv2.DIST_L2, maskSize = 0)

        # Sort distances w.r.t region for every pixel
        distances = np.sort(distances, axis = 2)

        # Grab distance to the border of nearest region
        d1, d2 = distances[:, :, 0], distances[:, :, 1]

        # Compute RHS of weight map and mask background pixels
        w = w0 * np.exp(-1 / (2 * sigma ** 2)  * (d1 + d2) ** 2) * (regions == background_class)

    else: # Only a single region present in the image
        w = np.zeros_like(mask)

    # Instantiate a matrix to hold class weights
    wc_x = np.zeros_like(mask)
    
    # Compute class weights for each pixel class (background, etc.)
    for pixel_class, weight in wc.items():
    
        wc_x[mask == pixel_class] = weight
    
    # Add them to the weight map
    w = w + wc_x
    
    return w

def _class_weights(mask):
    ''' Create a dictionary containing the classes in a mask,
        and their corresponding weights to balance their occurence
    '''
    
    wc = defaultdict()

    # Grab classes and their corresponding counts
    unique, counts = np.unique(mask, return_counts = True)

    # Convert counts to frequencies
    counts = counts / np.product(mask.shape)

    # Get max. counts
    max_count = max(counts)

    for val, count in zip(unique, counts):
        wc[val] = max_count / count
    
    return wc