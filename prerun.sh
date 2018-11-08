#!/bin/bash -l

printf "PYTHONPATH=$PYTHONPATH\n"
printf "Adding deeplab and slim to PYTHONPATH\n"

export PYTHONPATH=${PYTHONPATH}:$(pwd)/research
export PYTHONPATH=${PYTHONPATH}:$(pwd)/research/slim

printf "New PYTHONPATH=$PYTHONPATH\n"

source /opt/ds3/bin/activate
