# vit-small
{"arch": "vit_small", "patch_size": 16, "out_dim": 65536, "norm_last_layer": false, "warmup_teacher_temp": 0.04, "teacher_temp": 0.07, "warmup_teacher_temp_epochs": 30, "use_fp16": false, "weight_decay": 0.04, "weight_decay_end": 0.4, "clip_grad": 0, "batch_size_per_gpu": 64, "epochs": 800, "freeze_last_layer": 1, "lr": 0.0005, "warmup_epochs": 10, "min_lr": 1e-05, "global_crops_scale": [0.25, 1.0], "local_crops_scale": [0.05, 0.25], "local_crops_number": 10, "seed": 0, "num_workers": 10, "world_size": 16, "ngpus": 8, "nodes": 2, "optimizer": "adamw", "momentum_teacher": 0.996, "use_bn_in_head": false, "drop_path_rate": 0.1}

# vit-base args
{"arch": "vit_base", "patch_size": 16, "out_dim": 65536, "norm_last_layer": true, "warmup_teacher_temp": 0.04, "teacher_temp": 0.07, "warmup_teacher_temp_epochs": 50, "use_fp16": false, "weight_decay": 0.04, "weight_decay_end": 0.4, "clip_grad": 0.3, "batch_size_per_gpu": 32, "epochs": 400, "freeze_last_layer": 3, "lr": 0.00075, "warmup_epochs": 10, "min_lr": 2e-06, "global_crops_scale": [0.25, 1.0], "local_crops_scale": [0.05, 0.25], "local_crops_number": 10, "seed": 0, "num_workers": 10, "world_size": 32, "ngpus": 8, "nodes": 4, "optimizer": "adamw", "momentum_teacher": 0.996, "use_bn_in_head": false, "drop_path_rate": 0.1}

# vit-large - estimate
{"arch": "vit_large", "patch_size": 16, "out_dim": 65536, "norm_last_layer": true, "warmup_teacher_temp": 0.04, "teacher_temp": 0.07, "warmup_teacher_temp_epochs": 50, "use_fp16": false, "weight_decay": 0.04, "weight_decay_end": 0.4, "clip_grad": 0.3, "batch_size_per_gpu": 32, "epochs": 200, "freeze_last_layer": 3, "lr": 0.0005, "warmup_epochs": 10, "min_lr": 2e-06, "global_crops_scale": [0.25, 1.0], "local_crops_scale": [0.05, 0.25], "local_crops_number": 10, "seed": 0, "num_workers": 10, "world_size": 32, "ngpus": 8, "nodes": 4, "optimizer": "adamw", "momentum_teacher": 0.996, "use_bn_in_head": false, "drop_path_rate": 0.1}

cd ..; git clone https://github.com/weiyx16/dino.git; cd simmim_cond_feat; export WORKDIR=/mntdata/yx_results; $WORKDIR/azcopy copy "https://zeliuwestus2.blob.core.windows.net/aml/amldata/imagenet1000/val?sv=2019-02-02&ss=btqf&srt=sco&st=2021-01-02T16%3A47%3A22Z&se=2050-01-03T16%3A47%3A00Z&sp=rwdlacup&sig=gumCrZNdxZ9YlbXcUOBQ0oYa9zbBU4TJkfZ4uWB7Wus%3D" ./ --recursive; unzip -q train.zip -d train; cd ../dino

export NR=0; git pull; export tag=dino_large_300ep; export WORKDIR=/mntdata/yx_results; sudo chmod 777 -R /ze_workdir; touch /ze_workdir/suspend.txt; python -m torch.distributed.launch --nproc_per_node 8 --master_addr 10.3.40.231 --master_port 27426 --nnodes 8 --node_rank $NR main_dino.py --arch vit_large --teacher_temp 0.07 --warmup_teacher_temp_epochs 50 --clip_grad 0.3 --freeze_last_layer 3 --lr 0.001 --min_lr 1e-06 --global_crops_scale 0.25 1.0 --local_crops_scale 0.05 0.25 --batch_size_per_gpu 32 --epochs 300 --data_path ../simmim_cond_feat --output_dir $WORKDIR/output/simmlim/$tag --tag $tag --drop_path_rate 0.2

# evaluation:
python eval_linear.py --evaluate --arch vit_base --patch_size 16 --n_last_blocks 1 --avgpool_patchtokens true --data_path /path/to/imagenet/train


export OUT_PATH=$WORKDIR/output/simmlim/dino_large_300ep_lr3e4_wu20; touch ../suspend.txt; python -m torch.distributed.launch --nproc_per_node=8 eval_linear.py --arch vit_large --patch_size 16 --n_last_blocks 1 --avgpool_patchtokens true --data_path ../ --pretrained_weights $OUT_PATH/checkpoint.pth --output_dir $OUT_PATH/lincls/ --lr 1e-3 --batch_size_per_gpu 512

## ref esvit swin large w7 evaluation:
export OUT_PATH=$WORKDIR/output/simmlim/esvit_swin_large_150ep; touch ../suspend.txt; python -m torch.distributed.launch --nproc_per_node=8 eval_linear.py --data_path ../ --output_dir $OUT_PATH/lincls/ --pretrained_weights $OUT_PATH/checkpoint.pth --checkpoint_key teacher --batch_size_per_gpu 128 --arch swin_tiny --cfg experiments/imagenet/swin/swin_large_patch4_window7_224.yaml --n_last_blocks 4 --num_labels 1000 MODEL.NUM_CLASSES 0

# prepare val dataset
/msrhyper-weka/hanhu/azcopy copy "https://zeliuwestus2.blob.core.windows.net/aml/amldata/imagenet1000/val?sv=2019-02-02&ss=btqf&srt=sco&st=2021-01-02T16%3A47%3A22Z&se=2050-01-03T16%3A47%3A00Z&sp=rwdlacup&sig=gumCrZNdxZ9YlbXcUOBQ0oYa9zbBU4TJkfZ4uWB7Wus%3D" ./ --recursive