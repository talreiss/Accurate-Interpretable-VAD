#!/bin/bash

echo "Downloading CUHK-Avenue dataset....."

wget "http://101.32.75.151:8181/dataset/avenue.tar.gz"
tar -xvf avenue.tar.gz
rm avenue.tar.gz
rm avenue/avenue.mat
mv avenue.mat avenue/
python count_frames.py

echo "Download CUHK-Avenue successfully..."
