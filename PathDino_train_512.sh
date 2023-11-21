#!/bin/bash

# to run on 1 GPU
python PathDino_main_512.py \
                            --arch pathdino \
                            --lr 0.0005 \
                            --epochs 27 \
                            --batch_size_per_gpu 64 \
                            --data_path '/mayo_atlas/atlas/publicDatasets/TCGA_patches/histology_patches/' \
                            --output_dir /mayo_atlas/home/m288756/mayo_ai/output/DinoPath_512 \
                            --num_workers 24 \

# # to run on more than one GPUs
# python -m torch.distributed.launch --nproc_per_node=3  PathDino_main_512.py \
#                             --lr 0.0005 \
#                             --epochs 27 \
#                             --batch_size_per_gpu 64 \
#                             --data_path ['/mayo_atlas/home/m288756/mayo_ai/data/download_and_patch_for_training/histology_patches']
#                             --output_dir /mayo_atlas/home/m288756/mayo_ai/output/DinoPath_Vit_small_5blocks_same224Settings \
#                             --num_workers 24 \
#                             --host '28500' \