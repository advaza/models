#!/bin/bash -l

echo "PYTHONPATH=$PYTHONPATH\n"
echo "Adding deeplab and slim to PYTHONPATH\n"
export PYTHONPATH=${PYTHONPATH}:$(pwd)/research
export PYTHONPATH=${PYTHONPATH}:$(pwd)/research/slim
echo "New PYTHONPATH=$PYTHONPATH\n"

source /opt/ds3/bin/activate
