#!/bin/bash

echo "Downloading ShanghaiTech dataset....."

wget "http://101.32.75.151:8181/dataset/shanghaitech.tar.gz"
tar -xvf shanghaitech.tar.gz
rm shanghaitech.tar.gz
python count_frames.py

echo "Download ShanghaiTech successfully..."