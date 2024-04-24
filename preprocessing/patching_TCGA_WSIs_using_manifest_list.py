import cv2
import numpy as np
from openslide import OpenSlide
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from multiprocessing import Pool
from tqdm import tqdm
import os



def plot_thumbnail_and_mask(thumbnail, tissue_mask, patch_locations,  mask_hratio, mask_wratio):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the thumbnail
    ax[0].imshow(thumbnail)
    ax[0].set_title('Thumbnail')
    
    # Plot the patches on the thumbnail
    for (x, y) in patch_locations:
        rect = patches.Rectangle((x, y), mask_hratio, mask_wratio, linewidth=1, edgecolor='r', facecolor='none')
        ax[0].add_patch(rect)

    # Plot the tissue mask
    ax[1].imshow(tissue_mask, cmap='gray')
    ax[1].set_title('Tissue Mask')

    plt.savefig('plot.png')  # save the figure
    plt.show()


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


def get_patch_locations(tissue_mask,  mask_hratio, mask_wratio, tissue_threshold):
    contours, _ = cv2.findContours(tissue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    patch_locations = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w >= mask_wratio and h >= mask_hratio:
            for i in range(x, x + w - mask_wratio, mask_wratio):
                for j in range(y, y + h - mask_hratio, mask_hratio):
                    tissue_patch = tissue_mask[j:j + mask_hratio, i:i + mask_wratio]
                    # if np.sum(tissue_patch) / (mask_hratio ** 2) > tissue_threshold:
                    if np.count_nonzero(tissue_patch)/tissue_patch.size  >= tissue_threshold:
                        patch_locations.append((i, j))
    return patch_locations


def save_patch(patch, x, y, output_folder, wsi_name):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    patch_filename = f"{wsi_name}_{x}_{y}.jpg"
    # convert to RGB
    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    cv2.imwrite(str(output_folder / patch_filename), patch)



def process_wsi(wsi_path, output_folder, output_patch_size=1024, tissue_percent=0.9):
    try:
        wsi_obj = OpenSlide(wsi_path)
        
        wsi_name = Path(wsi_path).stem + ".svs"
        
        thumbnail = wsi_obj.get_thumbnail((1024, 1024))
        cthumbnail = clean_thumbnail(thumbnail)
        tissue_mask = ((cthumbnail.mean(axis=2) != 255) * 255).astype(np.uint8)
        
        w, h = wsi_obj.dimensions

        if 'openslide.objective-power' in wsi_obj.properties:
            objective_power = int(wsi_obj.properties['openslide.objective-power'])
        else:
            # objective_power = some_default_value  # Replace this with a suitable default value
            pixel_size = wsi_obj.level_downsamples[0]
            objective_power = np.round(np.log2(w / pixel_size) * 100)

        patch_size = (objective_power / 20.) * 1000
        mask_hratio = int((tissue_mask.shape[0] / h) * patch_size)
        mask_wratio = int((tissue_mask.shape[1] / w) * patch_size)

        Mask_to_WSI_ratioW = int(w / tissue_mask.shape[1])
        Mask_to_WSI_ratioH = int(h / tissue_mask.shape[0])
        
        patch_locations = get_patch_locations(tissue_mask, mask_hratio, mask_wratio, tissue_percent)

        min_distance = mask_hratio  # Minimum distance between points

        filtered_patch_locations = []
        for (x, y) in patch_locations:
            if is_far_enough((x, y), filtered_patch_locations, min_distance):
                filtered_patch_locations.append((x, y))

        plot_thumbnail_and_mask(cthumbnail, tissue_mask, filtered_patch_locations,  mask_hratio, mask_wratio)  # Plotting here
        
        scaled_patch_coordinates = []
        for (x, y) in filtered_patch_locations:
            scaled_patch_coordinates.append((int(x * Mask_to_WSI_ratioW), int(y * Mask_to_WSI_ratioH)))
            
        # Iterate through each patch location
        for (x, y) in scaled_patch_coordinates:
            # Extract the patch from the WSI
            patch_size_20x = int((objective_power/20.)*output_patch_size)
            patch = wsi_obj.read_region((x, y), 0, (patch_size_20x, patch_size_20x))
            if patch.size[0] != output_patch_size:
                patch = patch.resize((output_patch_size, output_patch_size))
            # Convert the patch to RGB and save it
            patch_rgb = np.array(patch.convert('RGB'))
            save_patch(patch_rgb, x, y, output_folder, wsi_name)
    except Exception as e:
        print(f"Error processing {wsi_path}: {e}")
        return False
        
def process_wsi_wrapper(args):
    return process_wsi(*args)

def process_wsi_directory(wsi_directory, output_folder, manifest_file, output_patch_size=1024, tissue_percent=0.9):
    
    # create the output folder
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # first find all the wsi files already patched
    total_wsi_set = set()
    def map_patches_to_wsi(folders):
        for dir_name in folders:
            # Loop over all files in the directory
            for filename in os.listdir(dir_name):
                if filename.endswith(".jpg"):
                    wsi_name = filename.split('.svs')[0]+'.svs'
                    # add to total_wsi_set
                    total_wsi_set.add(wsi_name)
    
    map_patches_to_wsi(["/TCGA/histology_patches/images", 
                       "/TCGA/histology_patches/images1024"])
    
    print(f"Found {len(total_wsi_set)} slides already patched")
    print(list(total_wsi_set)[0])
        
    # read the txt file
    with open(manifest_file) as f:
        lines = f.readlines()
    # pick up the second column which is the wsi name
    lines = [line.split('\t')[1] for line in lines]
    # remove the first line which is the header
    wsi_names = lines[1:]
    print(f"Found {len(wsi_names)} WSIs in the txt file")
    
    # filter out the wsi files that have already been patched
    wsi_names = [f'{wsi_directory}/{wsi_path}' for wsi_path in wsi_names if Path(wsi_path).name not in total_wsi_set]
    print(f"Found {len(wsi_names)} slides to process")
    
    args = [(wsi_path, output_folder, output_patch_size, tissue_percent) for wsi_path in wsi_names]
    
    with Pool(24) as pool:
        max_ = len(args)
        with tqdm(total=max_) as pbar:
            for i, _ in enumerate(pool.imap_unordered(process_wsi_wrapper, args)):
                pbar.update()
        
# Run the code
# the wsi directory is the one that contains the TCGA wsi files of the gdc_manifest_svs_Diagnostic_slide_Open_11765.txt list
wsi_directory = "/tcga_data/image/diagnostic"
# the output folder is the one that contains the patches supposing we already have two folders one for 512 size "images" and one for 1024 size "images1024"
output_folder = "/TCGA/histology_patches/images1024_2/"
# the manifest file is the one that contains the wsi names
manifest_file = "/preprocessing/gdc_manifest_svs_Diagnostic_slide_Open_11765.txt"

process_wsi_directory(wsi_directory, output_folder, manifest_file, output_patch_size=1024, tissue_percent=0.9)