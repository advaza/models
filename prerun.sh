#!/bin/bash -l

printf "PYTHONPATH=$PYTHONPATH\n"

echo 'export PYTHONPATH="/opt/ds3/bin/python"' >> ~/.bashrc
source ~/.bashrc

printf "New PYTHONPATH=$PYTHONPATH\n"
echo "pre run ok"




#printf "PYTHONPATH=$PYTHONPATH\n"
#printf "Adding deeplab and slim to PYTHONPATH\n"

#export PYTHONPATH=${PYTHONPATH}:$(pwd)/research
#export PYTHONPATH=${PYTHONPATH}:$(pwd)/research/slim

#printf "New PYTHONPATH=$PYTHONPATH\n"

#source /opt/ds3/bin/activate
