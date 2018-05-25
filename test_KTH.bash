#!/bin/bash

python test_KTH.py  \
    --name=kth_recursive\
    --K=${1} \
    --F=${1} \
    --T=${2} \
    --data=KTH --dataroot=${3} --textroot=videolist/KTH/ --pick_mode=Slide \
    --image_size 128 \
