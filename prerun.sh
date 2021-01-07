#!/usr/bin/env bash

apt-get update
apt-get install -y protobuf-compiler ffmpeg libsm6 libxext6
cd /cnvrg/research
protoc object_detection/protos/*.proto --python_out=.
cd /cnvrg
printf "machine api.wandb.ai\n  login user\n  password ${WANDB_TOKEN}\n" > /root/.netrc
