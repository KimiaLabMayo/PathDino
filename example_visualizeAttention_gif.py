# import streamlit as st
import os
import argparse
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision
from PIL import Image
import torch.nn as nn
import numpy as np
import cv2

from PathDino import get_pathDino_model


# Load PathDino model and image transforms
model, _ = get_pathDino_model("./inference/PathDino512.pth")



def visualize_attention_ViT(model, img, patch_size=16):
        attention_list = []
        # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # device = torch.device("cpu")
        
        print("Image shape:", img.shape)
    
        w_featmap = img.shape[-2] // patch_size
        h_featmap = img.shape[-1] // patch_size
        attentions = model.get_last_selfattention(img.to(torch.device("cpu")))
        nh = attentions.shape[1] # number of head
        # we keep only the output patch attention
        attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
        attentions = attentions.reshape(nh, w_featmap, h_featmap)
        attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].detach().numpy()
        for j in range(nh):
            attention_list.append(attentions[j])
        return attention_list
    
def cross_fade(image1_path, image2_path, steps):
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    
    print(type(image1), type(image2))
    print(image1_path)
    print(image2_path)
    
    # Resize images to match if necessary
    if image1.shape != image2.shape:
        h, w = image2.shape[:2]
        image1 = cv2.resize(image1, (w, h))

    # Generate cross-fade images
    return [cv2.addWeighted(image1, float(steps - i) / steps, image2, float(i) / steps, 0) for i in range(steps + 1)]

def create_gif_from_images(new_crop, input_image, output_dir, attention_list):
    basename = os.path.basename(input_image).split('.')[0]
    print(basename)
    # Number of steps for the transition
    steps = 20

    for i in range(0, len(attention_list)):
        # read the activated attention map
        image2_path = os.path.join(output_dir, basename +"_attn-head" + str(i) + ".png")

        # Generate cross-fade images
        print(input_image, image2_path)
        images = cross_fade(new_crop, image2_path, steps)
        # Convert OpenCV images to PIL images for saving to gif
        images = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in images]

        # Save as gif with the name of the attention head
        images[0].save(os.path.join(output_dir,basename +"_attn-head" + str(i) + ".gif"), save_all=True, append_images=images[1:], loop=0, duration=100)

# Define the function to generate activation maps
def generate_activation_maps(input_image, output_dir="./output", patch_size=16):
    # Load the image and apply preprocessing
    img_PIL = Image.open(input_image).convert('RGB')
    basename = os.path.basename(input_image).split('.')[0]

    # Convert the image to a NumPy array
    img = np.array(img_PIL)
    
    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[0] % patch_size, img.shape[1] - img.shape[1] % patch_size
    min_size = min(w, h)
    print("Image shape:", img.shape)
    preprocess = transforms.Compose([
            transforms.Resize((img.shape[0], img.shape[1])),
            transforms.CenterCrop((w, h)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the tensors
        ])
    
    # resize the PIL image to (img.shape[0], img.shape[1])
    PIL_image = img_PIL.resize((img.shape[1], img.shape[0]))
    # crop the PIL image to (w, h) from center
    PIL_image = PIL_image.crop(((img.shape[1] - w) // 2,
                                (img.shape[0] - h) // 2,
                                (img.shape[1] + w) // 2,
                                (img.shape[0] + h) // 2))
    # PIL_image = PIL_image.crop((0, 0, h, w))
     
    # print("Image shape:", img_PIL.shape)
    image_tensor = preprocess(img_PIL)
    img = image_tensor.unsqueeze(0).to(torch.device("cpu"))
    # Generate activation maps
    with torch.no_grad():
        attention_list = visualize_attention_ViT(model=model, img=img, patch_size=16)

    # save attentions heatmaps
    os.makedirs(output_dir, exist_ok=True)
   # save the PIL image as RGB
    PIL_image = PIL_image.convert('RGB')
    PIL_image.save(os.path.join(output_dir, "img_new.png"))
    torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True), os.path.join(args.output_dir, "img.png"))
    for j in range(len(attention_list)):
        fname = os.path.join(output_dir, basename +"_attn-head" + str(j) + ".png")
        plt.imsave(fname=fname, arr=attention_list[j], format='png')
        print(f"{fname} saved.")
        
    # create gif from attention heatmaps
    create_gif_from_images(os.path.join(output_dir, "img_new.png"), input_image, output_dir, attention_list)
    print("GIFs saved.")
    return attention_list

def main(args):
    # Load PathDino model and image transforms
    model, _ = get_pathDino_model("./inference/PathDino512.pth")

    # Generate activation maps
    attention_list = generate_activation_maps(args.input_image, args.output_dir, args.patch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate activation maps and attention gifs using PathDino model.")
    parser.add_argument("input_image", type=str, help="Path to the input image.")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory for saving results.")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size for activation maps.")

    args = parser.parse_args()
    main(args)