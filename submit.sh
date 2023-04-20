#!/bin/sh
#BSUB -q gpua100
#BSUB -J Classifier
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 02:00
#BSUB -R "rusage[mem=8GB]"
##BSUB -R "select[gpu40gb]" #options gpu40gb or gpu80gb
#BSUB -o outputs/gpu_%J.out
#BSUB -e outputs/gpu_%J.err
# -- end of LSF options --

nvidia-smi

source aml/bin/activate

# Options
# Run main.py --help to get options


python3 main.py --name resnet18 --batch_size 24 --max-epochs 100 --num-workers 16 >| outputs/test.out 2>| error/test.err