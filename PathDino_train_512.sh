#!/bin/bash

# to run on 1 GPU
python PathDino_main_512.py \
                            --arch pathdino \
                            --lr 0.0005 \
                            --epochs 27 \
                            --batch_size_per_gpu 64 \
                            --data_path ['/path/to/histology_patches/'] \
                            --output_dir /output/dir \
                            --num_workers 24 \

# # to run on more than one GPUs
# python -m torch.distributed.launch --nproc_per_node=3  PathDino_main_512.py \
#                             --lr 0.0005 \
#                             --epochs 27 \
#                             --batch_size_per_gpu 64 \
#                             --data_path ['/path/to/histology_patches']
#                             --output_dir /output/dir \
#                             --num_workers 24 \
#                             --host '28500' \
