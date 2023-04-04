#! /bin/bash
echo "Dataset name"
read data_name

echo "Object detection features"
python pre_processing/bboxes.py --dataset_name=$data_name --train
python pre_processing/bboxes.py --dataset_name=$data_name

echo "Optical flow features"
python pre_processing/flows.py --dataset_name=$data_name --train
python pre_processing/flows.py --dataset_name=$data_name

echo "Velocity and deep representation fearures"
python feature_extraction.py --dataset_name=$data_name