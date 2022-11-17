## Data download and Preparation
To properly download and prepare the data, please change your directory to:
```
cd path-to-directory/Interpretable_VAD/data/
```

To download and prepare each evaluation dataset, please refer to the corresponding dataset section.
### 1. Ped2
Execute the following command to download and prepare the Ped2 dataset:
```
sh ped2.sh
```
### 2. CUHK Avenue
Execute the following command to download and prepare the Avenue dataset:
```
sh avenue.sh
```
### 3. ShanghaiTech
Execute the following command to download the ShanghaiTech dataset:
```
sh shanghaitech.sh
```
The ShanghaiTech dataset contains only raw videos as training data. For extraction of the frames, run the following command: 
```
python extract_shanghaitech_frames.py
```
Test frames are already provided.