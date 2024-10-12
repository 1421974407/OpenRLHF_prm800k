#!/bin/bash

#SBATCH -p llmit6
#SBATCH --quotatype=reserved
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH -J zhaojian
#SBATCH --output=./%A.out

# set -e

# Initialize Conda environment
# source $HOME/.bashrc
# eval "$(conda shell.bash hook)"
source activate
cd $HOME/zhaojian/OpenRLHF_prm800k
conda activate zhaojian_prm

# source $HOME/myenv.sh

# export PATH=$PATH:/mnt/petrelfs/share/test-cuda/cuda-12.1/bin
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/petrelfs/share/test-cuda/cuda-12.1/lib64
# echo $PATH
# echo $LD_LIBRARY_PATH
# echo $CUDA_VISIBLE_DEVICES
# n_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
# echo "n_gpus: $n_gpus"
# #unset CUDA_VISIBLE_DEVICES
# #echo $CUDA_VISIBLE_DEVICES
nvidia-smi
GPU_IDS=$CUDA_VISIBLE_DEVICES
echo $CUDA_VISIBLE_DEVICES

# debug
# python openrlhf/cli/train_prm.py --save_path ./checkpoint/mistal-7b-prm --save_steps 500 --logging_steps 1 --eval_steps 100 --train_batch_size 256 --micro_train_batch_size 8 --pretrain mistralai/Mistral-7B-v0.1  --bf16 --max_epochs 1 --max_len 8192 --zero_stage 3 --learning_rate 1e-6 --dataset peiyi9979/Math-Shepherd --input_key input --label_key label --flash_attn --load_checkpoint --gradient_checkpointing --packing_samples --wandb_group prm --placeholder_token "ки" --reward_tokens "+" "-"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
deepspeed --include="localhost:${GPU_IDS}" --module openrlhf.cli.train_prm \
   --save_path ./checkpoint/llama-2-7b-tiny-dataset \
   --save_steps 500 \
   --logging_steps 1 \
   --eval_steps 100 \
   --train_batch_size 256 \
   --micro_train_batch_size 2 \
   --pretrain meta-llama/Llama-2-7b-hf  \
   --bf16 \
   --max_epochs 1 \
   --max_len 8192 \
   --zero_stage 2 \
   --learning_rate 1e-6 \
   --dataset Birchlabs/openai-prm800k-stepwise-critic \
   --input_key instruction,responses,next_response \
   --label_key rating \
   --flash_attn \
   --load_checkpoint \
   --gradient_checkpointing \
   --packing_samples \
   --wandb_group prm \
   --placeholder_token "ки" \
   --reward_tokens "+" "-"