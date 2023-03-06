## Attribute-based Representations for Accurate and Interpretable Video Anomaly Detection

Official PyTorch Implementation of [**"Attribute-based Representations for Accurate and Interpretable Video Anomaly Detection"**](https://arxiv.org/pdf/2212.00789.pdf). 

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/attribute-based-representations-for-accurate/anomaly-detection-on-shanghaitech)](https://paperswithcode.com/sota/anomaly-detection-on-shanghaitech?p=attribute-based-representations-for-accurate)
	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/attribute-based-representations-for-accurate/anomaly-detection-on-chuk-avenue)](https://paperswithcode.com/sota/anomaly-detection-on-chuk-avenue?p=attribute-based-representations-for-accurate)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/attribute-based-representations-for-accurate/abnormal-event-detection-in-video-on-ucsd)](https://paperswithcode.com/sota/abnormal-event-detection-in-video-on-ucsd?p=attribute-based-representations-for-accurate)

![interpretable VAD](./figures/interpretability.png)

## 1. Dependencies
```
python -m venv venv 
source venv/bin/activate
pip install -r requirements.txt
```
Download this [file](https://drive.google.com/file/d/1fxMmmZ8TmmdGOovbC2QYPgWTaRxyVdE0/view?usp=sharing) (195 MB npy file) and place it in the following path: ```./data/shanghaitech/train/pose.npy```.

## 2. Usage
### 2.1 Data download and Preparation
To download the evaluation datasets, please follow the [instructions](./data/README.md).

### 2.2 and 2.3 Data preparation and Feature Extraction 
For extracting velocity and deep representations, run the following command:
```
sh data_preprocessing.sh
```

### 2.4 Score calibration
To compute calibration parameters for each representation, run the following command:
```
python score_calibration.py [--dataset_name]
```
### 2.4 Evaluation
Finally, you can evaluate by running the following command:
```
python evaluate.py [--dataset_name] [--sigma]
```
We usually use ```--sigma=3``` for Ped2 and Avenue, and ```--sigma=7``` for ShanghaiTech.

You can download our set of representations for Ped2, Avenue and ShanghaiTech datasets from [here](https://drive.google.com/drive/folders/1vSMpDb5jIyc2tNJaYVphguUlFcwPayms?usp=sharing).

## 3. Results

| UCSD Ped2 | CUHK Avenue | ShanghaiTech |
|:---------:|:-----------:|:------------:|
|   99.1%   |    93.6%    |    85.9%     |


## Citation
If you find this useful, please cite our paper:
```
@article{reiss2022attribute,
  title={Attribute-based Representations for Accurate and Interpretable Video Anomaly Detection},
  author={Reiss, Tal and Hoshen, Yedid},
  journal={arXiv preprint arXiv:2212.00789},
  year={2022}
}
```
