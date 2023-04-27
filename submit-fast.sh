#!/bin/sh
#BSUB -q gpua100
#BSUB -J fast2
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -W 03:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "select[gpu80gb]" #options gpu40gb or gpu80gb
#BSUB -o outputs/gpu_%J.out
#BSUB -e outputs/gpu_%J.err
#BSUB -N
# -- end of LSF options --

nvidia-smi

source ../envs/3d/bin/activate

# Options
# Run main.py --help to get options


CUDA_LAUNCH_BLOCKING=1 python3 main.py --name resnet18_compiled_deepspeed2 --batch_size 50 --max-epochs 100 --num-workers 16 --fast --num_devices -1 --compiled >| outputs/fast_compiled2.out 2>| error/fast_compiled2.err
