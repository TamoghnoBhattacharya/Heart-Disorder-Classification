# HEARTBEAT CLASSIFICATION FOR WEARABLES
Entry for Hackoff v3.0 Hackathon in the **Siemens Healthineers Challenge** - ECG Classification of Heart Disorders.
The same project is also applicable for the Open track **Data Science and AI** .
<br>
## Team: ***Gray Matter***
- Arpit Agarwal
- Tamoghno Bhattacharya
- Varun Bhardwaj
- Smitesh Hadape
- Hardik Bhati

<br>

## Problem Statement
We were told to propose and implement an intelligent real time heartbeat classification algorithm based on ECG graph data.<br>
The goal of the study was to design a model that is able to classify cardiac arrhythmia 
(17 diagnostic classes encompassing “normal sinus rhythm”, “pacemaker rhythm” and 15 other rhythm disorders) 
effectively from analysis of long-duration (10-s) ECG signal fragments.<br><br>
We implemented a modified version of the paper ["Arrhythmia detection using deep convolutional neural network with 
long duration ECG signals"](https://www.sciencedirect.com/science/article/pii/S0010482518302713 "Link to paper").

<br>

## Dataset
The dataset was acquired from [here](https://data.mendeley.com/datasets/7dybx7wyfn/3).
ECG signals were obtained from the [PhysioNet service ](http://www.physionet.org) from the **MIT-BIH Arrhythmia** database. 
- The ECG signals were from 45 patients: 19 female (age: 23-89) and 26 male (age: 32-89). 
- The ECG signals contained 17 classes: normal sinus rhythm, pacemaker rhythm, and 15 types of cardiac dysfunctions (for each of which at least 10 signal fragments were collected). 
- All ECG signals were recorded at a sampling frequency of 360 [Hz] and a gain of 200 [adu / mV]. 
- For the analysis, 1000, 10-second (3600 samples) fragments of the ECG signal (not overlapping) were randomly selected.
- Only signals derived from one lead, the MLII, were used. 
- Original data is in mat format (Matlab).
<br>

## Approach
### ***Preprocessing (Normalization)***
- Firstly, we wrote a function to convert .mat files to .csv files and combined them to form a dataset which was used for further purpose.
- Data was preprocessed which included rescaling the ECG data values between **[-1,1]** so as to have a better and a faster classification.

### ***Classification Model***
- We tried our hands on a variety of methods to find which one was best suited for the given task and found that 1-D CNN model worked best.
- Since, the number of classes were more (17) and the dataset was comparatively smaller, we had to use only training and testing sets.
- The proposed 1D CNN Classification model consists of 6-7 convolutional layers having dropout after each of them along with ‘relu’ activation. 
- After flattening, the last layer consists of a ‘softmax’ layer. 
- Many changes had to be done by brute-force inorder to get a high accuracy model.

### ***Deployment***
- Our model can be used in clinical scenarios along with the use in Wearables.
- The patient ECG can be aquired and sent through the mobile phone to the cloud where our developed model is trained and kept.
- Result can be validated by ECG beats and the message regarding the defect/disease will be received on mobile device as well as Wearables.
<br>

## Result
At the end of 20 epochs for the 17-classes the training and validation stages attained accuracy rates of 98% and 93.6%, respectively.
The result could be sometimes abnormal as the dataset is small and many of the classes have just 10-11 fragments, so training over them couldn’t provide apt results.
Also, we tried using LSTM, RCNN, but 1-D CNN was the simplest and optimized compared to all other models.


## References
**[`https://www.frontiersin.org/articles/10.3389/fphys.2020.569050/full`](https://www.frontiersin.org/articles/10.3389/fphys.2020.569050/full)**
