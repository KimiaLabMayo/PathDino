
from FPS_helper import process_wsi
import openslide
import numpy as np
import random
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_features_fromMosaic_fastWSIR(mosaic, model, transform_input, batch_size):
    input_batch =torch.stack([transform_input(patch) for patch in mosaic])
    return model(input_batch.to(device)).detach().cpu().numpy()


def get_bob_FPS(slide_path, tissue_threshold, returnSamples, network, transform_input, batch_size):
    
    slide = openslide.open_slide(slide_path)
    returnSamples = 30
    output_patch_size = 1000
        # mosaic, objective_power = process_wsi(slide_path,output_patch_size=1000, tissue_percent=0.7, returnSamples=returnSamples, mask_output_folder="", saveMask=True, returnMasks=False)
    # mosaic, objective_power = process_wsi(slide_path,output_patch_size=1000, tissue_percent=0.5, returnSamples=returnSamples, mask_output_folder="", saveMask=True, returnMasks=False)
    mosaic, objective_power = process_wsi(slide_path,output_patch_size=1000, tissue_percent=tissue_threshold, returnSamples=returnSamples, mask_output_folder="", saveMask=False, returnMasks=False)
    
    print('mosaic', len(mosaic))
    final_patches = []
    # Iterate through each patch location
    for (x, y) in mosaic:
        # Extract the patch from the WSI
        patch_size_20x = int((objective_power/20.)*output_patch_size)
        patch = slide.read_region((x, y), 0, (patch_size_20x, patch_size_20x)).convert('RGB')
        final_patches.append(patch)
    # shuffle the patches
    random.shuffle(final_patches)
    final_patches = final_patches[:returnSamples]

    print(f'{slide_path}, Number of patches: ', len(mosaic))
    features = extract_features_fromMosaic_fastWSIR(final_patches, network, transform_input, batch_size)

    # print(np.array(features).shape)
    bob_raw = (np.diff(np.array(features), axis=1) < 0)*1
    if len(bob_raw) == 0:
        return None, None
    bob = BoB(bob_raw)
    return bob, features



    
network = model.to(device)
returnSamples = 30
bob, features = get_bob_FPS(slide_path, tissue_threshold, returnSamples, network, transform_input, batch_size)
