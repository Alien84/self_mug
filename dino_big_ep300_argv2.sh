#!/bin/bash
# source /.bashrc

# These 3 lines are necessary for running .sh files on nori, thorin etc. No need for running .sh files through condor
export PATH="/jmain02/home/J2AD011/hxc06/aaa85-hxc06/anaconda3/bin:$PATH"
source /jmain02/home/J2AD011/hxc06/aaa85-hxc06/anaconda3/bin/activate
conda activate env-06

NODES=1
NGPUS=8
ARCH=deit_small
PATCH=16
OUTDIM=8192
BS=64
EPOCHS=300
WORKER=10
LAM=1
MU=0.001
NU=0.0
BETA=0.0
DATA="/jmain02/home/J2AD011/hxc06/shared/imagenet/train/"
OUT="./checkpoints/exp1_300_argv2"

cd /jmain02/home/J2AD011/hxc06/aaa85-hxc06/code/mugs_similarity_indexRevised/
python run_with_submitit.py --partition 'big' --timeout 1440 --nodes $NODES --ngpus $NGPUS \
        --data_path $DATA \
	--output_dir $OUT \
	--arch vit_small \
	--instance_queue_size 65536 \
	--local_group_queue_size 65536 \
	--use_bn_in_head false \
	--instance_out_dim 256 \
	--instance_temp 0.2 \
	--local_group_out_dim 256 \
	--local_group_temp 0.2 \
	--local_group_knn_top_n 8 \
	--group_out_dim 65536 \
	--group_student_temp 0.1 \
	--group_warmup_teacher_temp 0.04 \
	--group_teacher_temp 0.07 \
	--group_warmup_teacher_temp_epochs 30 \
	--norm_last_layer false \
	--norm_before_pred true \
	--batch_size_per_gpu $BS \
	--epochs 300 \
	--warmup_epochs 10 \
	--clip_grad 3.0 \
	--lr 0.0008 \
	--min_lr 1e-06 \
	--patch_embed_lr_mult 0.2 \
	--drop_path_rate 0.1 \
	--weight_decay 0.04 \
	--weight_decay_end 0.1 \
	--freeze_last_layer 1 \
	--momentum_teacher 0.996 \
	--use_fp16 false \
	--local_crops_number 10 \
	--size_crops 96 \
	--global_crops_scale 0.25 1 \
	--local_crops_scale 0.05 0.25 \
	--timm_auto_augment_par rand-m9-mstd0.5-inc1 \
	--prob 0.5 \
	--use_prefetcher true \
	--debug false
