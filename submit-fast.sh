#!/bin/sh
#BSUB -q gpua100
#BSUB -J deepspeed
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 06:00
#BSUB -R "rusage[mem=6GB]"
#BSUB -R "select[gpu80gb]" #options gpu40gb or gpu80gb
#BSUB -o outputs/gpu_%J.out
#BSUB -e outputs/gpu_%J.err
#BSUB -B
#BSUB -N
# -- end of LSF options --

nvidia-smi

source ../envs/3d/bin/activate

# Options
# Run main.py --help to get options


CUDA_LAUNCH_BLOCKING=1 python3 main.py --name resnet18_fast --batch_size 24 --max-epochs 100 --num-workers 16 --fast >| outputs/fast.out 2>| error/fast.err
