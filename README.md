# Intrusion-Detection-System-using-Multi-classification-Algorithm

## Introduction:

_In the present modern world, information are transferred through network. Cyber Attacks on Network significantly threatens the data privacy of an individual as well as organizations worldwide. As cyber threats continue to evolve, traditional security systems seems to be lacking to detect and mitigate new network attacks and these attacks are evolving rapidly. This project presents a An AI-Based Multi-Class Classification Approach for Detecting and Categorizing Complex Network Attacks in High-Dimensional Traffic Using CICIDS2017, capable of identifying and categorizing modern network threats with high accuracy. The project begins with exploring dataset and the dataset that utilized in the project is CICIDS 2017 dataset which contains 79 features with a label feature indicating the attack type. The dataset contains some empty values which will be removed by using dropna() function in the data preprocessing part. The dataset contains various classes of attacks and there are some imbalance in the class, so we used SMOTE to oversample the dataset which makes the dataset to have equal amount of class distribution. we implement multiple classification algorithms—including XGBoost, Random Forest, Linear SVC, Naïve Bayes, D-Tree, and LightGBM to analyze and categorize normal it is. We analysed how efficiently these models detects attacks based on how Accurate , Precise, recall, F1-score, and computational efficiency to determine the most effective approach for threat detection. We used cross validation method which is Stratified K-Fold method which is used to make sure our models don’t overfit and underfit during the training process. After training and testing the models, we used Matplot function to visualize the scores, confusion matrix, learning curve of the models. In order to make our project more efficient we used Undersampler to our dataset and also compared it with our main model. The results of our project shows a great accuracy in detecting the Intrusion with the accuracy score of 99.89% for our Rf-GBM Algorithm .We found the potential of machine learning in adding some value to the cybersecurity domain, By providing an adaptive and adequate solution to counteract emerging network threats. This project enhances the development of intelligent Intrusion Detection System frameworks that can strengthen cybersecurity defences in modern network infrastructures_

## Objectives:

The main objective of this project is to develop an advanced Classification ML-based intrusion detection system to detect complex network attacks using multiple classification algorithms and the CICIDS 2017 dataset. By addressing the limitations of existing methods, this study aims to enhance attack detection accuracy by training models with best parameters, manage class imbalance by introducing feature selection, and improve the real-world effectiveness of NIDS in securing modern networks.
_•	Develop an Enhanced AI Based network attack detection model using various Multi classification algorithm to improve detection of new evolving modern network threats and to detect unknown network attacks that are emerging rapidly .
•	To develop the advanced NI Detection System with the top performing Multi classification algorithms like XGBoost, LightGBM, Random Forest, Decision Tree, Naïve Bayes, Linear SVC etc .
•	Overfitting can cause the model to detect the Network attack correctly, so by using K-Fold cross-validation stratified we overcame this overfitting problem.
•	To balance the used dataset in training the model. Unbalanced dataset can make the model inefficient , so we used SMOTE module to synthetically generate the data and to balance all class features in the dataset which are nothing but Network attacks._

So in this project we've used Multi-Classification algorithms to build a model that is more precise in detecting intrusion in any Network.

## Models that we've used in this project:
•	XGBoost
• Random Forest
• Linear SVC
• Naïve Bayes
• D-Tree 
• LightGBM 

### I've uploaded the code to train the IDS model. 

# Thank You
