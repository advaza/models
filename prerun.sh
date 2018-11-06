#!/bin/bash

export PYTHONPATH=${PYTHONPATH}:$(pwd)/research
export PYTHONPATH=${PYTHONPATH}:$(pwd)/research/slim

python3 research/deeplab/train.py --train_logdir output/mobile_net --model_variant=mobilenet_v2 --tf_initial_checkpoint=checkpoints/mobilenet_v2_1.0_224/mobilenet_v2_1.0_224.ckpt --train_split=train_aug --dataset_dir=/data/aug_pascal_tfrecord/tfrecord
