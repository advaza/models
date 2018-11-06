#!/bin/bash

export PYTHONPATH=${PYTHONPATH}:$(pwd)/research
export PYTHONPATH=${PYTHONPATH}:$(pwd)/research/slim

source /opt/ds3/bin/activate

python3 research/deeplab/train.py --train_logdir output/mobile_net_params_from_yswang0522 --fine_tune_batch_norm=True --model_variant=mobilenet_v2 --tf_initial_checkpoint=checkpoints/mobilenet_v2_1.0_224/mobilenet_v2_1.0_224.ckpt --train_split=train_aug --dataset_dir=/data/aug_pascal_tfrecord/tfrecord --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=513 --train_crop_size=513 --num_clones=4 --train_batch_size=16
