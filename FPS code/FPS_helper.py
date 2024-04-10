import cv2
import numpy as np
from openslide import OpenSlide
from pathlib import Path
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from multiprocessing import Pool
from tqdm import tqdm
import os
import pandas as pd
import random
from sklearn.neighbors import KernelDensity

def get_sampled_points_density_proportional_KDE(points, desired_sample_size):
    num_points = len(points)
    if num_points <= desired_sample_size:
        return points

    points_arr = np.array(points)
    
    # Fit KDE model to the points
    kde = KernelDensity(bandwidth=0.1)  # You can adjust the bandwidth
    kde.fit(points_arr)

    # Generate samples from the KDE
    samples = kde.sample(desired_sample_size)
    final_sample = samples.tolist()

    return final_sample


def RGB2HSD(X):
    eps = np.finfo(float).eps
    X[np.where(X==0.0)] = eps
    
    OD = -np.log(X / 1.0)
    D  = np.mean(OD,3)
    D[np.where(D==0.0)] = eps
    
    cx = OD[:,:,:,0] / (D) - 1.0
    cy = (OD[:,:,:,1]-OD[:,:,:,2]) / (np.sqrt(3.0)*D)
    
    D = np.expand_dims(D,3)
    cx = np.expand_dims(cx,3)
    cy = np.expand_dims(cy,3)
            
    X_HSD = np.concatenate((D,cx,cy),3)
    return X_HSD


def clean_thumbnail(thumbnail):
    thumbnail_arr = np.asarray(thumbnail)
    
    wthumbnail = np.zeros_like(thumbnail_arr)
    wthumbnail[:, :, :] = thumbnail_arr[:, :, :]

    thumbnail_std = np.std(wthumbnail, axis=2)
    wthumbnail[thumbnail_std<5] = (np.ones((1,3), dtype="uint8")*255)
    thumbnail_HSD = RGB2HSD( np.array([wthumbnail.astype('float32')/255.]) )[0]
    kernel = np.ones((30,30),np.float32)/900
    thumbnail_HSD_mean = cv2.filter2D(thumbnail_HSD[:,:,2],-1,kernel)
    wthumbnail[thumbnail_HSD_mean<0.05] = (np.ones((1,3),dtype="uint8")*255)
    return wthumbnail

                
def is_far_enough(new_point, existing_points, min_distance):
    for point in existing_points:
        if np.sqrt((new_point[0] - point[0])**2 + (new_point[1] - point[1])**2) < min_distance:
            return False
    return True


def get_patch_locations(tissue_mask, cthumbnail,  mask_hratio, mask_wratio, tissue_threshold):
    contours, mm = cv2.findContours(tissue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_with_contours = cthumbnail.copy()
    cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)  # Draw contours on the image
    
    image_with_rectangles = cthumbnail.copy()
    
    patch_locations = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # plot the rectangles on the image_with_rectangles
        cv2.rectangle(image_with_rectangles, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        if w >= mask_wratio and h >= mask_hratio:
            for i in range(x, x + w - mask_wratio, mask_wratio):
                for j in range(y, y + h - mask_hratio, mask_hratio):
                    tissue_patch = tissue_mask[j:j + mask_hratio, i:i + mask_wratio]
                    # if np.sum(tissue_patch) / (mask_hratio ** 2) > tissue_threshold:
                    tissue_magnitude = np.count_nonzero(tissue_patch)/tissue_patch.size
                    if tissue_magnitude  >= tissue_threshold:
                        patch_locations.append(((i, j),tissue_magnitude))

    return patch_locations, image_with_contours, image_with_rectangles

def process_wsi(wsi_path, output_patch_size=1000, tissue_percent=0.9, returnSamples=30, mask_output_folder="/home/m288756/mayo_ai/src/WSIretrieval/Output/CRC/masks/", saveMask=False, returnMasks=True):
    # print(wsi_path)
    # try:
        wsi_obj = OpenSlide(wsi_path)
        wsi_name = Path(wsi_path).stem + ".svs"
        
        thumbnail = wsi_obj.get_thumbnail((1024, 1024))
        cthumbnail = clean_thumbnail(thumbnail)
        tissue_mask = ((cthumbnail.mean(axis=2) != 255) * 255).astype(np.uint8)
        
        try:
            objective_power = int(wsi_obj.properties['openslide.objective-power'])
        except:
            objective_power = 20
             
        w, h = wsi_obj.dimensions
        patch_size = (objective_power / 20.) * 1000
        mask_hratio = int((tissue_mask.shape[0] / h) * patch_size)
        mask_wratio = int((tissue_mask.shape[1] / w) * patch_size)
        # estimate the mask patch size given the size of the WSI, the size of the mask, and the output patch size
        mask_patch_size = int(output_patch_size / mask_wratio)
       
        
        Mask_to_WSI_ratioW = int(w / tissue_mask.shape[1])
        Mask_to_WSI_ratioH = int(h / tissue_mask.shape[0])
        
        patch_locations, image_with_contours, image_with_rectangles = get_patch_locations(tissue_mask, cthumbnail, mask_hratio, mask_wratio, tissue_percent)
        
        min_distance = mask_hratio * 2  # Minimum distance between points

        filtered_patch_locations = []
        for (x, y), _ in patch_locations:
            if is_far_enough((x, y), filtered_patch_locations, min_distance):
                filtered_patch_locations.append((x, y))

        filtered_patch_locations = get_sampled_points_density_proportional_KDE(filtered_patch_locations, returnSamples)
        
        
        # if returnMasks or saveMask:
        #     mask = plot_thumbnail_and_mask(cthumbnail,image_with_contours,image_with_rectangles, tissue_mask, filtered_patch_locations,  mask_hratio, mask_wratio, wsi_name, mask_output_folder,saveMask)  # Plotting here

        scaled_patch_coordinates = []
        for (x, y) in filtered_patch_locations:
            scaled_patch_coordinates.append((int(x * Mask_to_WSI_ratioW), int(y * Mask_to_WSI_ratioH)))

        if returnMasks:
            return scaled_patch_coordinates,objective_power, mask
        else:
            return scaled_patch_coordinates, objective_power
