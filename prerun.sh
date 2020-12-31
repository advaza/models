#!/usr/bin/env bash

apt install -y protobuf-compiler
cd /cnvrg/research
protoc object_detection/protos/*.proto --python_out=.
cd /cnvrg
