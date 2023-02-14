# PoseFusion: PoseFusion: Robust Object-in-Hand Pose Estimation with SelectRNN
[Project website:https://elevenjiang1.github.io/ObjectInHand-Dataset/](https://elevenjiang1.github.io/ObjectInHand-Dataset/)



## 1. Data Extraction
We upload all our dataset in [GoogleDrive](), all the data are saved in rosbag format. 
And example code for data extraction can be available from DataExtraction Folder.


### 1.1 Usage
(1) Build all package
```
git clone git@github.com:elevenjiang1/ObjectInHand-Dataaset.git
cd ObjectInHand-Dataset/DataExtraction
catkin_make
```

(2) Run rosbag and see data
```
# Always need to publish static transform
roscd data_extraction/scripts/files
rosbag play -l tf_static.bag

# Rosbag play data
cd /file/to/Download/cracker
rosbag play rosbag play 2022-08-14-16-00-16-processed.bag
```


(3) Record data
```
# Data will saved in 
roscd data_extraction/scripts/
python extract_data.py
```
