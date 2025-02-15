## An Attribute-based Method for Video Anomaly Detection (TMLR)

Official PyTorch Implementation of [**"An Attribute-based Method for Video Anomaly Detection
"**](https://arxiv.org/pdf/2212.00789.pdf). 

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/attribute-based-representations-for-accurate/anomaly-detection-on-shanghaitech)](https://paperswithcode.com/sota/anomaly-detection-on-shanghaitech?p=attribute-based-representations-for-accurate)
	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/attribute-based-representations-for-accurate/anomaly-detection-on-chuk-avenue)](https://paperswithcode.com/sota/anomaly-detection-on-chuk-avenue?p=attribute-based-representations-for-accurate)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/attribute-based-representations-for-accurate/abnormal-event-detection-in-video-on-ucsd)](https://paperswithcode.com/sota/abnormal-event-detection-in-video-on-ucsd?p=attribute-based-representations-for-accurate)

![interpretable VAD](./figures/interpretability.png)

## 1. Dependencies
```
python==3.7
torch==1.12.0+cu102
```
Download this [file](https://drive.google.com/file/d/1fxMmmZ8TmmdGOovbC2QYPgWTaRxyVdE0/view?usp=sharing) (195 MB npy file) and place it in the following path: ```./data/shanghaitech/train/pose.npy```.
## 2. Usage
### 2.1 Data download and Preparation
To download the evaluation datasets, please follow the [instructions](./data/README.md).

### 2.2 Data preparation
The training and testing datasets should be preprocessed according to the these [instructions](./pre_processing/README.md).

### 2.3 Feature Extraction
For extracting velocity and deep representations, run the following command:

```
python feature_extraction.py [--dataset_name]
```

Pose representations have already been provided by us.

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
|   99.1%   |    93.7%    |    85.9%     |


## Citation
If you find this useful, please cite our paper:
```
@article{reissattribute,
  title={An Attribute-based Method for Video Anomaly Detection},
  author={Reiss, Tal and Hoshen, Yedid},
  journal={Transactions on Machine Learning Research}
}
```
