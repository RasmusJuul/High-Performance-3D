#!/bin/sh
#BSUB -q gpua100
#BSUB -J compile
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 01:00
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


# CUDA_LAUNCH_BLOCKING=1 python3 main.py --name baseline16 --batch_size 16 --max-epochs 100 --num-workers 16 >| outputs/baseline.out 2>| error/baseline.err

CUDA_LAUNCH_BLOCKING=1 python3 main.py --name compiled16 --batch_size 16 --max-epochs 100 --num-workers 16 --fast --compiled >| outputs/compiled.out 2>| error/compiled.err

# CUDA_LAUNCH_BLOCKING=1 python3 main.py --name deepspeed16 --batch_size 16 --max-epochs 100 --num-workers 16 --fast --deepspeed >| outputs/deepspeed.out 2>| error/deepspeed.err

# CUDA_LAUNCH_BLOCKING=1 python3 main.py --name compiled_deepspeed16 --batch_size 16 --max-epochs 100 --num-workers 16 --fast --compiled --deepspeed >| outputs/compiled-deepspeed.out 2>| error/compiled-deepspeed.err