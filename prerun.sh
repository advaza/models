#!/bin/bash

echo "PYTHONPATH=$PYTHONPATH"
echo "Adding deeplab and slim to PYTHONPATH"
export PYTHONPATH=${PYTHONPATH}:$(pwd)/research
export PYTHONPATH=${PYTHONPATH}:$(pwd)/research/slim
echo "New PYTHONPATH=$PYTHONPATH"

source /opt/ds3/bin/activate
