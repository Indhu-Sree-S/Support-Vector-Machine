# Support-Vector-Machine

Problem Statement:
	Analyze the prior marketing campaigns of a Bank using SVM and predict if the user will buy the Bank’s term deposit or not
	
Problem Description:
	Initially preprocess and clean the data. Visualize the data for better understanding. Then build SVM model using linear kernel and Radial basis kernel. Tune and compare both the models for the best result.
	
Code:
	Visualizing the dependent variable ‘y’ using countplot. 
#EDA
	It shows that the data type of each feature in the data.
	It shows that the data has no null value.
#preprocessing
#significant variable
	Correlations between the variables are shown above.
	Significant variables are determined by chi-square test. Insignificant variables are identified by p-value, which is greater than 0.1. So those variables can be removed for further process. Marital, default, day, balance are removed from the data.
#feature scaling
#validation split
#linear_svc
	Performed Cross validation for the linear SVC model with 10 folds and each fold had equal number of instances supporting each customer class. To get the better performance result, we perform hyper parameter tuning.
Hyper Parameter Tuning Results
Best Params :  {'C': 1}
Best Score :  1.0
Best Model :  SVC(C=1, kernel='linear', probability=True)
	Random search technique shows that the model classifies the customer with C as 1.
CLASSIFICATION REPORT OF VALIDATION DATA
              precision    recall  f1-score   support
           0       1.00      1.00      1.00       401
           1       1.00      1.00      1.00        52
    accuracy                           1.00       453
   macro avg       1.00      1.00      1.00       453
weighted avg       1.00      1.00      1.00       453

CONFUSION MATRIX OF VALIDATION DATA
[[401   0]
 [  0  52]]
Acuuracy :  1.0
AUC score :  1.0
	Above model has 100% precision and recall, shows that the precision quantifies the number of positive class predictions that actually belong to the positive class and Recall quantifies the number of positive class predictions made out of all positive examples in the dataset. It has an accuracy of 100% shows that the class prediction is good. AUC (Area under ROC curve) score provides an aggregate measure of performance across all possible classification thresholds, thus it shows that this model’s performance are 100% good. Thus, the linear kernel SVC fitted to predict whether the customer will subscribe a term deposit has come with good results.

#RBF kernel SVC
	Performed Cross validation for the RBF SVC model with 10 folds and each fold had equal number of instances supporting each customer class. To get the better performance result, we perform hyper parameter tuning.
Hyper Parameter Tuning Results
Best Params :  {'C': 1}
Best Score :  1.0
Best Model :  SVC(C=1, probability=True)
	Random search technique shows that the model classifies the customer with C as 1.

CLASSIFICATION REPORT OF VALIDATION DATA
              precision    recall  f1-score   support
           0       1.00      1.00      1.00       401
           1       1.00      1.00      1.00        52
    accuracy                           1.00       453
   macro avg       1.00      1.00      1.00       453
weighted avg       1.00      1.00      1.00       453

CONFUSION MATRIX OF VALIDATION DATA
[[401   0]
 [  0  52]]
Acuuracy :  1.0
AUC score :  1.0
	Above model has 100% precision and recall, shows that the precision quantifies the number of positive class predictions that actually belong to the positive class and Recall quantifies the number of positive class predictions made out of all positive examples in the dataset. It has an accuracy of 100% shows that the class prediction is good. AUC (Area under ROC curve) score provides an aggregate measure of performance across all possible classification thresholds, thus it shows that this model’s performance are 100% good. Thus, the RBF SVC fitted to predict whether the customer will subscribe a term deposit has come with good results.

Comparison:
	Both Linear SVC and RBF SVC have a 100% in precision, recall, accuracy, AUC. It shows that both models to predict whether the customer will subscribe a term deposit or not.
Conclusion:
	Both the models are best at its works, we can use any model to predict whether the customer will subscribe a term deposit or not. By analyzing the bank marketing, it shows that marketing will effective only on the customer who subscribed the deposit in the bank.

