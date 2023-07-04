#!/bin/sh

#SBATCH --time=2-00:00:00
#SBATCH -N 1 # number of node
#SBATCH -n 1
#SBATCH -p cas_v100_4
#SBATCH --comment python
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=40

## Jarvis
#srun python train_folder.py --root_dir '../dataset/' --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 --dataset 'Jarvis' --property 'fe'  --epochs 1000 --batch_size 64 --resume 0
#srun python train_folder.py --root_dir '../dataset/' --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 --dataset 'Jarvis' --property 'total_energy'  --epochs 1000 --batch_size 64 --resume 0
#srun python train_folder.py --root_dir '../dataset/' --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 --dataset 'Jarvis' --property 'opt_bandgap'  --epochs 1000 --batch_size 64 --resume 0
#srun python train_folder.py --root_dir '../dataset/' --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 --dataset 'Jarvis' --property 'mbj_bandgap'  --epochs 1000 --batch_size 64 --resume 0
#srun python train_folder.py --root_dir '../dataset/' --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 --dataset 'Jarvis' --property 'bulk_modulus_kv'  --epochs 1000 --batch_size 64
#srun python train_folder.py --root_dir '../dataset/' --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 --dataset 'Jarvis' --property 'shear_modulus_gv'  --epochs 1000 --batch_size 64 --resume 0

## MP
#srun python train_folder.py --root_dir '../dataset/' --n_train 60000 --n_val 5000 --n_test 4132 --dataset 'MP' --property 'formation_energy'  --epochs 1000 --batch_size 64 --resume 0
#srun python train_folder.py --root_dir '../dataset/' --n_train 60000 --n_val 5000 --n_test 4132 --dataset 'MP' --property 'band_gap'  --epochs 1000 --batch_size 64 --resume 0
srun python train_folder.py --root_dir '../dataset/' --n_train 4664 --n_val 393 --n_test 393 --dataset 'MP' --property 'bulk'  --epochs 1000 --batch_size 64 --resume 0
#srun python train_folder.py --root_dir '../dataset/' --n_train 4664 --n_val 393 --n_test 393 --dataset 'MP' --property 'shear'  --epochs 1000 --batch_size 64 --resume 0