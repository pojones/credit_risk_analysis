# Credit Risk Analysis
## Overview
This project focuses on the implementation of various machine learning models to determine credit risk; an inherently unbalanced classification problem, by utilizing imbalanced-learn and scikit-learn libraries.  A fictional peer-to-peer lending services company “LendingClub” has generated a dataset with credit card information. Using the RandomOverSampler and SMOTE algorithms, and undersampling the data using the ClusterCentroids algorithm will produce the first results. Further processing of the data will include the SMOTEENN algorithm to produce a combination of over- and under-sampling results. These results are then fed to two machine learning models for comparison; BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict the credit risk values. These two models are then compared and analyzed to predict credit risk/worthiness. 
## Results
### Logistic Regression
Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. For this reason, a logistic regression classification model works well to predict credit risk. To extract more accurate results, the training data was resampled to include more instances of bad loans using the following four methods:

#### Naïve Random Oversampling
-	Balanced Accuracy: 0.652
-	Precision High Risk: 0.01
-	Precision Average: 0.99
-	Recall High Risk: 0.62
-	Recall Average: 0.68
<br/>

![](https://github.com/pojones/credit_risk_analysis/blob/f845b2f9f75eec333ec777048881da143c52a8ee/images/naiveOversamplingResults.png)

#### SMOTE Oversampling
-	Balanced Accuracy: 0.624
-	Precision High Risk: 0.01
-	Precision Average: 0.99
-	Recall High Risk: 0.59
-	Recall Average: 0.66
<br/>

![](https://github.com/pojones/credit_risk_analysis/blob/f845b2f9f75eec333ec777048881da143c52a8ee/images/smoteOversamplingResults.png)

#### Cluster Centroids Undersampling
-	Balanced Accuracy: 0.516
-	Precision High Risk: 0.01
-	Precision Average: 0.99
-	Recall High Risk: 0.60
-	Recall Average: 0.44
<br/>

![](https://github.com/pojones/credit_risk_analysis/blob/f845b2f9f75eec333ec777048881da143c52a8ee/images/undersamplingResults.png)

#### Combination Sampling (Over and Under with SMOTEENN)
-	Balanced Accuracy: 0.62
-	Precision High Risk: 0.01
-	Precision Average: 0.99
-	Recall High Risk: 0.71
-	Recall Average: 0.53
<br/>

![](https://github.com/pojones/credit_risk_analysis/blob/f845b2f9f75eec333ec777048881da143c52a8ee/images/combinationSampling.png)

### Ensemble Classifiers
The data was also fed to two machine learning models with anticipation of increasing the model’s performance.
#### Balanced Random Forest
-	Balanced Accuracy: 0.789
-	Precision High Risk: 0.03
-	Precision Average: 0.99
-	Recall High Risk: 0.70
-	Recall Average: 0.87
<br/>

![](https://github.com/pojones/credit_risk_analysis/blob/f845b2f9f75eec333ec777048881da143c52a8ee/images/randomForestResults.png)

#### Easy Ensemble AdaBoost
-	Balanced Accuracy: 0.93
-	Precision High Risk: 0.09
-	Precision Average: 0.99
-	Recall High Risk: 0.92
-	Recall Average: 0.94
<br/>

![](https://github.com/pojones/credit_risk_analysis/blob/f845b2f9f75eec333ec777048881da143c52a8ee/images/adaBoostResults.png)

## Summary
The Easy Ensemble AdaBoost Classifier had the highest accuracy at 93%. However, this is not always the best case in an unbalanced dataset. The model’s accuracy can be attributed to how well it fits for low-risk applicants. However, risk assessment is generally focused on the higher risk applicants, since they would be the clients with the greatest liability. 
All the models had low precision for high-risk, most of them exhibiting extremely low precision (5%). It is probably a good thing to keep this number low, since it represents how many erroneously classified high-risk applicants there are. 
The Easy Ensemble AdaBoost classifier had the highest recall for high-risk applicants at 94%. This model is likely the best at detecting high risk applicants by a reasonable amount. 
Looking at the data, the Easy Ensemble AdaBoost classifier seems to have yielded the best results. It is good at detecting high risk applicants despite having the most erroneous high-risk classifications. The majority of the applicants are low-risk, so this model makes the most sense for this application. However, these results leave room for further analyses of the high-risk candidates. 
