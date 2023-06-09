#!/bin/bash

#SBATCH -p gpu5
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres gpu:2
#SBATCH -o train%j.out # 注意可以修改"slurm"为与任务相关的内容方便以后查询实验结果
#SBATCH --mem 40G

date
python finetune.py
date