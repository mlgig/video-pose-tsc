# Interpretable Classification of Human Exercise Videos through Pose Estimation and Multivariate Time Series Analysis

This repository contains code and data for the paper [Interpretable Classification of Human Exercise Videos through Pose Estimation and Multivariate Time Series Analysis.](https://www.researchgate.net/publication/348436597_Interpretable_Classification_of_Human_Exercise_Videos_through_Pose_Estimation_and_Multivariate_Time_Series_Analysis), accepted as a full research paper at [5th International Workshop on Health Intelligence
(W3PHIAI-21) at AAAI21](http://w3phiai2021.w3phi.com/index.html#).
The paper presents an approach for the classification and interpretation
of human motion from video data. It applies state-of-the-art advances in
the area of Human Pose Estimation for video and Multivariate Time Series Classification
and Interpretation. The aim is to facilitate physiotherapists, coaches and rehabilitation
patients by providing feedback after the execution of physical exercises. 

The data used are pre-recorded videos of the Military Press exercise. The first step consists of applying
OpenPose to extract the time series data from the video. The second step
classifies the extracted data using multivariate time series classification methods (MTSC). 
We chose deep learning methods FCN and Resnet for classification and 1NN-DTW as a baseline. 
MTSC methods such as MrSEQL, FCN, Resnet also support interpretation 
by highlighting the discriminative region. We further compared the results
with the highly efficient ROCKET classifier, which also works well for multivariate time series.

![Alt text](figs/overview.png?raw=true)
<em>**Fig 1** shows the overview of the proposed approach for the Military Press exercise. Going from raw video to
extracting and tracking body points using human pose estimation, and preparing the resulting data for time series 
classification and interpretation.</em>

## Data Description
The data used is the video recordings of the execution of the Military Press (MP) exercise. Participants completed 10 
repetitions of the normal form and 10 repetitions of induced forms. The data folder consists of the extracted time 
series data which is already splitted into training/test using the 70/30 split. The folder TrainTestData consists of 
data in the numpy format whereas the folder TrainTestDataSktime consists data in the [sktime](https://www.sktime.org/en/latest/) format. 
The data is further resampled to the max length for all the samples. There are roughly 1300 and 600 samples in training 
and testing data respectively. Each data sample is a multivariate time series data with a shape of 161x8 (161 length 
and 8 dimensions). Each dimension corresponds to the location of a given body part in a frame. 

### Installation
Please use the requirements.txt file to install all the dependencies. There is a configuration script for each 
classifier script which contains the relative paths to the exercise and data folders.

```python
# All scripts follow the same structure
# python script_name.py --exercise_config path_to_exercise_config --knn_config path_to_knn_config 
```


## Results
Classifier Name | Accuracy (Unnormalized data) | Accuracy (Normalized data)
--------------- | -----------------------------| ---------------
1NN-DTW | 0.58 | 0.50 
ROCKET | **0.81** | **0.68**
FCN | **0.72** | **0.65**
Resnet | **0.73** | **0.65**

The above table shows the average accuracy on test data over three train/test splits. Normalising the time series
significantly reduces the accuracy of all classifiers, due to losing information about the range and magnitude of the 
signal capturing the exercise movement. This points out the importance of time series classifiers that can also work 
work with unnonrmalised data.

## Visualization
We further use the Class Activation Mapping (CAM) explanation approach to find the discriminative region for a given 
time series. CAM provides a vector of importance weights for each point in the time series, a higher CAM value means a 
higher discriminative value for the classifier. The discriminative region is mapped back to the original frames in the 
video. For privacy concerns we haven't made the videos public. The figure below shows the discriminative region and the
corresponding frames for FCN-CAM on class A. The 3 frames extracted are the ones corresponding to the top 3 highest values of CAM for 
this time series.

![Alt](figs/fcn_region2.jpg) ![Alt](figs/fcn_frame.png)

<em>**Fig 2** shows the discriminative region (left) and the corresponding frames (right) 
for class A for FCN-CAM. </em>

## Citation
Please cite this paper as:
```
@incollection{singh2021interpretable,
  title={Interpretable Classification of Human Exercise Videos through Pose Estimation and Multivariate Time Series Analysis},
  author={Singh, Ashish and Le, Binh Thanh and Le Nguyen, Thach and Whelan, Darragh and O’Reilly, Martin and Caulfield, Brian and Ifrim, Georgiana},
  year={2021},
  booktitle = {5th International Workshop on Health Intelligence(W3PHIAI-21) at AAAI21},
  publisher={Springer}
}
```

