This repository contains code and data for the paper [Interpretable Classification of Human Exercise Videos through Pose Estimation and Multivariate Time Series Analysis.](https://www.researchgate.net/publication/348436597_Interpretable_Classification_of_Human_Exercise_Videos_through_Pose_Estimation_and_Multivariate_Time_Series_Analysis)
The paper presents an approach for the classification and interpretation
of human motion from video data. It applies state-of-the-art advances in
the area of Human Pose Estimation for video and Multivariate Time Series Classification
and Interpretation. The aim is to facilitate physiotherapists, coaches and rehabilitation
patients by providing feedback after the execution of physical exercises. 

The data used is the pre-recorded videos of the Military Press exercise. The first step consists of applying
OpenPose to extract the time series data from the video. The second step
classifies the extracted data using multivariate time series classification methods (MTSC). 
We chose deep learning methods FCN and Resnet for classification and KNN (with n=1) as a baseline. 
MTSC methods such as MrSEQL, FCN, Resnet also support interpretation 
by highlighting the discriminative region. We further compared the results with 
with the highly efficient ROCKET classifier, which also works well for multivariate time series.

![Alt text](figs/overview.png?raw=true)
<em>**Fig 1** shows the overview of the proposed approach for the Military Press exercise. Going from raw video to
extracting and tracking body points using human pose estimation, and preparing the resulting data for
time series classification and interpretation.</em>

## Data Description
The data used is in the form of video recordings of the execution of the Military Press (MP) exercise.
Participants completed 10 repetitions of the normal form and 10 repetitions of induced forms. 
The data folder consist of the extracted time series data which is already splitted into training/test using the 70/30 split.
The folder data/TrainTestData consists of data in the numpy format whereas the folder data/TrainTestDataSktime consists data in the [sktime](https://www.sktime.org/en/latest/) format.
The data is further resampled to have same length for all the samples.
There are roughly 1300 and 600 samples in training
and testing data respectively. Each data sample is a multivariate time series data with a
shape of 161x8 (161 length and 8 dimensions). Each dimension corresponds to the location of a given body part in a frame.

### Installation
Please use the requirements.txt file to install all the dependencies. There is a configuration script for each 
classifier script which contains the relative paths to the exercise and data folders.


## Results
Classifier Name | Accuracy (Unnormalized data) | Accuracy (Normalized data)
--------------- | -----------------------------| ---------------
1NN-DTW | 0.58 | 0.50 
ROCKET | **0.81** | **0.68**
FCN | **0.72** | **0.65**
Resnet | **0.73** | **0.65**

Table showing average accuracy on test data over three train/test splits. Normalising the time series
significantly reduces the accuracy of all classifiers, due to losing information about the range and
magnitude of the signal capturing the exercise movement.

## Visualization
We use the Class Activation Mapping (CAM) XAI approach to find the discriminative region for a given time series.
The discriminative region is mapped back to the original frames in the video. The figure below 
shows the discriminative region and the corresponding frames for
FCN. The frames are taken from regions marked with a red ellipse.

![Alt](figs/fcn_region2.jpg) ![Alt](figs/fcn_frame.png)

<em>**Fig 2** shows the discriminative region (left) and the corresponding frames (right) 
for class A for FCN-CAM. </em>

## Citation
Please cite this paper as:
'''
@incollection{singh2021interpretable,
  title={Interpretable Classification of Human Exercise Videos through Pose Estimation and Multivariate Time Series Analysis},
  author={Singh, Ashish and Le, Binh Thanh and Le Nguyen, Thach and Whelan, Darragh and O’Reilly, Martin and Caulfield, Brian and Ifrim, Georgiana},
  year={2021},
  booktitle = {5th International Workshop on Health Intelligence(W3PHIAI-21) at AAAI21},
  publisher={Springer}
}
'''

