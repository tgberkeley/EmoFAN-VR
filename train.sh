#!/bin/bash
#SBATCH --partition=gpgpuC,gpgpu,gpgpuB
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=tg220 # required to send email notifcations - please replace <your_username> with your college login name or email address
#SBATCH --output=/vol/bitbucket/tg220/results/out_test_practice%j.out
export PYTHONPATH=/vol/bitbucket/tg220/emotiondetector/emonet_master
export PATH=/vol/bitbucket/tg220/miniconda3/bin/:$PATH
source activate
source /vol/cuda/10.0.130/setup.sh
TERM=vt100 # or TERM=xterm
#/usr/bin/nvidia-smi
#uptime
echo "Beginning training..."
python -u /vol/bitbucket/tg220/emotiondetector/emonet_master/train.py
