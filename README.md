# RTA-Accident_Severity-Classification-Deployment
# Road Traffic Severity Classification🚦
![Python](https://img.shields.io/badge/Language-python3.9-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XgBoost-orange)
![Frontend](https://img.shields.io/badge/Framework-Streamlit-red)
![Deployment](https://img.shields.io/badge/Cloud-Streamlit-purple)

## Introduction
With the increasing of road traffic infrastructures, motor vehicles, drivers, and traffic flow, the role of road traffic in supporting and guiding economic and social development is becoming more and more obvious. As a result, road traffic safety has increasingly become a key issue in concerning the safety of people’s lives and property, as well as affecting the quality and efficiency of economic and social development. Road traffic accidents are the process of simultaneous damage of people or things, which caused by the coupling imbalance of dynamic and static factors such as human, vehicle, road, and environment. Therefore, it is necessary to study the influencing factors, as well as the classification and identification model of the severity of road traffic accident, so as to pave the way for improving the safety level of road traffic.

## 🧭 Problem Statement: 
This is a multi-class classification problem where we are predicting the severity of accident :
* Slight Injury
* Fatal Injury
* Serious Injury

based on the other 31 features.

## 🧾 Description: 
This data set is collected from **Addis Ababa Sub-city Police Departments** for master's research work. The data set has been prepared from manual records of road traffic accidents of the year 2017-20. All the sensitive information has been excluded during data encoding and finally it has 32 features and 12316 instances of the accident.

### :bar_chart: Exploratory Data Analysis:
* Exploratory Data Analysis is the first step of understanding your data and acquiring domain knowledge. 

### :hourglass: Data Preprocessing:
* The dataset has around 16 features with missing values. This missing values are imputed using **Predictive Imputation** technique where I used the known values to predict the missing values.

### ⚖ Handeling Data Imbalance:
* The Dataset was quite imbalanced with 10415 records with Slight injury, 1743 records withSerious injury anf just 158 records with Fatal injury.
* I used **SMOTE** method for balancing the dataset. 

### :mag_right: Features Selection:
* On using **model.feature_importances_**.

### ⚙ Model Training:
* On training my model using several classification algorithms, the model trained with **ExtraTreesClassifier** gave best results. 
* Used **RepeatedStratifiedKFold** with 5 splits cross validation with hyper-parameter tuning on ExtraTreesClassifier (baseline model) using **GridSearchCV**.
* Also, I found that my baseline model (ExtraTreesClassifier) was overfitting the dataset. On investigation I found that the dataset was affected by **Curse of Dimensionality**. So I reduced the dimensions and trained my model again.
* After retraining my model, I found that it was generalizing well with an accuracy of **92.23%**.
* As per the problem statement I used **F1 Score** as the evaluation metric for my model.

## Web Application :computer: :earth_americas: : 
Built a web application using Streamlit and deployed on Streamlit cloud.
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://abhissaro-rta-accident-severity-classification-app-n6s9il.streamlit.app/)
