#!/bin/bash 
#SBATCH --partition=contrib-gpuq 
#SBATCH --qos=gpu 
#SBATCH --job-name=QuadraNet_v2_small_detection
#SBATCH --output=gpu_basics.%j.out 
#SBATCH --error=gpu_basics.%j.out 
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100.80gb:4
#SBATCH --mem=256GB
#SBATCH --export=ALL 
#SBATCH --time=5-00:00:00 
#SBATCH --cpus-per-task=32
#
set echo 
umask 0022 
# to see ID and state of GPUs assigned
export LD_LIBRARY_PATH=/usr/lib64
nvidia-smi 

## Load the necessary modules
module load gnu10
module load python
cd /home/cxu21/QuadraNet_v2/QuadraNet/object_detection

bash dist_train.sh configs/quadranet/cascade_mask_rcnn_quadranet_small_7x7_3x_coco_in1k.py 4 --cfg-options model.pretrained=/home/cxu21/QuadraNet_v2/QuadraNet/logs/QuadraNet_v2_rbf_small_ImageNet_with_inputprojection/checkpoint-best.pth