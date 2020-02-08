#!/bin/bash -l

#$ -S /bin/bash
#$ -l h_rt=72:00:0
#$ -pe mpi 1
#$ -l mem=16G
#$ -N asasrec2
#$ -wd /home/ucacjm1/workspace/Adversarial-Collaborative-Filtering/
#source /home/ucacjm1/anaconda3/bin/activate /home/ucacjm1/anaconda3/envs/keras/

conda activate keras
python /home/ucacjm1/workspace/Adversarial-Collaborative-Filtering/run_adv_ori.py --path /home/ucacjm1/workspace/Adversarial-Collaborative-Filtering/ --model $1 --dataset $2 --epochs 1000 --adv_epoch 0 --verbose 20 --embed_size $3 --opath janEval/new_asasrec2_dense_pos/ --eval_mode all --reg_adv $4 --eps $5 --eps_pos $6 --eps_dense $7 --eps_conv $8
