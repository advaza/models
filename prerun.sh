#!/bin/bash -l

printf "Adding deeplab and slim to PYTHONPATH\n"

echo 'export PYTHONPATH=${PYTHONPATH}:$(pwd)/research' >> ~/.bashrc
echo 'export PYTHONPATH=${PYTHONPATH}:$(pwd)/research/slim' >> ~/.bashrc

#source /opt/ds3/bin/activate
