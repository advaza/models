#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:"/cnvrg"

git clone https://github.com/advaza/research-clothes-segmentation.git && cd research-clothes-segmentation && git checkout feature/vis && cd .. && cp -r research-clothes-segmentation/visualizer . && pip install pydensecrf
