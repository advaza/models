#!/bin/bash -l

printf "PYTHONPATH=$PYTHONPATH\n"
printf "Adding deeplab and slim to PYTHONPATH\n"

echo 'export PYTHONPATH=${PYTHONPATH}:$(pwd)/research' >> ~/.bashrc
echo 'export PYTHONPATH=${PYTHONPATH}:$(pwd)/research/slim' >> ~/.bashrc

#source ~/.bashrc

printf "New PYTHONPATH=$PYTHONPATH\n"

#source /opt/ds3/bin/activate


