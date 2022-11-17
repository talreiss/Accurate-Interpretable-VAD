## Pre-processing Stage
To properly perform the pre-processing stage, please change your directory to:
```
cd path-to-directory/Interpretable_VAD/
```

### 1. Object Detection

Our object detector outputs are provided [here](https://drive.google.com/drive/folders/1BnjzuwxyXio2sNU_4w7rlTw4PcURlq_R?usp=sharing).
Set up the bounding boxes by placing the corresponding files in the following folders: 
- All files for Ped2 should be placed in: ``` ./data/ped2```
- All files for Avenue should be placed in: ``` ./data/avenue```
- All files for ShanghaiTech should be placed in: ``` ./data/shanghaitech```
----
This section describes how to prepare the object detector to extract bounding boxes:

Please install the Detectron2 library by executing the following commands:
```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
Then download the ResNet50-FPN weights by executing:
```
wget https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl -P pre_processing/checkpoints/
```

Run the following command to detect all the foreground objects. 
```
python pre_processing/bboxes.py [--dataset_name] [Optional: --train] 
```
E.g., In order to extract all train objects from Ped2:
```
python pre_processing/bboxes.py --dataset_name=ped2 --train 
```
This will save the results to `./data/ped2/ped2_bboxes_train.npy`,  where each item contains all the bounding boxes in a single video frame.

In order to extract all test objects from Ped2:
```
python pre_processing/bboxes.py --dataset_name=ped2
```
This will save the results to `./data/ped2/ped2_bboxes_test.npy`,  where each item contains all the bounding boxes in a single video frame.

### 2. Optical Flow
We extract optical flows in videos using use [FlowNet2.0](https://github.com/NVIDIA/flownet2-pytorch).
In order to install FlowNet2 please execute the following commands:
```
cd pre_processing
bash install_flownet2.sh
cd ..
```

First, download the pre-trained FlowNet2 weights (i.e., `FlowNet2_checkpoint.pth.tar`) from [here](https://drive.google.com/file/d/1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da/view?usp=sharing) 
and place it in `Interpretable_VAD/pre_processing/checkpoints/`.
Now, Run the following command to estimate all the optical flows:
```
python pre_processing/flows.py [--dataset_name] [Optional: --train] 
```
E.g., In order to extract flows from Ped2 train frames:
```
python pre_processing/flows.py --dataset_name=ped2 --train
```
This will save the results to `./data/ped2/training/flows/`,  where each item contains all the bounding boxes in a single video frame.

In order to extract flows from Ped2 test frames:
```
python pre_processing/flows.py --dataset_name=ped2
```
This will save the results to `./data/ped2/testing/flows/`,  where each item contains all the bounding boxes in a single video frame.