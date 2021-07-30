# Support-Vector-Machine
Problem Statement:
	Analyze the prior marketing campaigns of a Bank using SVM and predict if the user will buy the Bank’s term deposit or not
Problem Description:
	Initially preprocess and clean the data. Visualize the data for better understanding. Then build SVM model using linear kernel and Radial basis kernel. Tune and compare both the models for the best result.
Code:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
data = pd.read_csv("Bank",sep=',')
data.head()
sns.countplot(x='y',data=data)
Output:
 
	Visualizing the dependent variable ‘y’ using countplot. 
	
#EDA
data.dtypes
 
	It shows that the data type of each feature in the data.
print(data.isnull().sum())
 
	It shows that the data has no null value.

#preprocessing
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
data.iloc[:,1]=le.fit_transform(data.iloc[:,1])
data.iloc[:,2]=le.fit_transform(data.iloc[:,2])
data.iloc[:,3]=le.fit_transform(data.iloc[:,3])
data.iloc[:,4]=le.fit_transform(data.iloc[:,4])
data.iloc[:,6]=le.fit_transform(data.iloc[:,6])
data.iloc[:,7]=le.fit_transform(data.iloc[:,7])
data.iloc[:,8]=le.fit_transform(data.iloc[:,8])
data.iloc[:,10]=le.fit_transform(data.iloc[:,10])
data.iloc[:,-2]=le.fit_transform(data.iloc[:,-2])
data.iloc[:,-1]=le.fit_transform(data.iloc[:,-1])
data.head()
#significant variable
plt.figure(figsize=(12,10))
cor = data.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()
print(cor)
 
	Correlations between the variables are shown above.

X = data.iloc[:,:-1]  
y = data.iloc[:,-1]
from sklearn.preprocessing import MinMaxScaler
minmaxer = MinMaxScaler(feature_range=(1,10))
minmaxed_x = minmaxer.fit_transform(X)
from sklearn.feature_selection import chi2
chi_value,pval = chi2(minmaxed_x,y)
pval = np.round(pval,decimals=3)
with np.printoptions(precision=4,suppress=True):
 print(pd.DataFrame(np.concatenate((chi_value.reshape(-1,1),pval.reshape(-1,1)),axis=1),
 index = data.columns[:-1],columns=['chi2 val','pval']))
 
	Significant variables are determined by chi-square test. Insignificant variables are identified by p-value, which is greater than 0.1. So those variables can be removed for further process. Marital, default, day, balance are removed from the data.
#feature scaling
x = data.drop(columns=['marital','default',’day’,'balance']).values
y = data['y'].values
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler.fit_transform(x)

#validation split
from sklearn.model_selection import StratifiedShuffleSplit
def train_val_splitter(x,y):
 splitter = StratifiedShuffleSplit(n_splits=1,test_size=0.1,random_state=101)
 for train_ind,val_ind in splitter.split(x,y):
  xtrain = x[train_ind];xval = x[val_ind]
  ytrain = y[train_ind];yval = y[val_ind]
 return xtrain,xval,ytrain,yval
xtrain,xval,ytrain,yval = train_val_splitter(x,y)
#linear_svc
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
lin_svm = SVC(kernel='linear',probability=True)
splitter = StratifiedShuffleSplit(n_splits=10,random_state=101,test_size=0.2)
cv = cross_validate(estimator=lin_svm,X=xtrain,y=ytrain,cv=splitter,n_jobs=-1,scoring='precision',
 return_estimator=True,return_train_score=True)
plt.plot(range(len(cv['test_score'])),cv['test_score'],label='test score',color='crimson')
plt.plot(range(len(cv['train_score'])),cv['train_score'],label='train score')
plt.scatter(range(len(cv['test_score'])),cv['test_score'],color='crimson')
plt.scatter(range(len(cv['train_score'])),cv['train_score'])
plt.xlabel('Model')
plt.ylabel('Precision')
plt.xticks(range(len(cv['test_score'])))
plt.legend()
plt.title('Cross Validation Results')
plt.show()
best_lin_svm = cv['estimator'][0]
best_lin_svm
from sklearn.model_selection import RandomizedSearchCV
params={'C':np.arange(1,5)}
tuner = RandomizedSearchCV(estimator=best_lin_svm,param_distributions=params,n_jobs=-1,scoring='precision',
 cv=splitter,random_state=101,return_train_score=True,)
tuner.fit(xtrain,ytrain)
print("Hyper Parameter Tuning Results")
print("Best Params : ",tuner.best_params_)
print("Best Score : ",tuner.best_score_)
print("Best Model : ",tuner.best_estimator_)
best_lin_svm = tuner.best_estimator_
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score,roc_curve
ypred = best_lin_svm.predict(xval)
ytrue = yval
print("CLASSIFICATION REPORT OF VALIDATION DATA")
print(classification_report(ytrue,ypred))
print("CONFUSION MATRIX OF VALIDATION DATA")
conf_mat = confusion_matrix(ytrue,ypred)
print(conf_mat)
print("Acuuracy : ",(conf_mat[0][0]+conf_mat[1][1])/len(ytrue))
probs = best_lin_svm.predict_proba(xval)[:,1]
fpr,tpr,_ = roc_curve(yval,probs)
random_probs = [0 for i in range(len(yval))]
p_fpr,p_tpr,_ = roc_curve(yval,random_probs)
auc_score = roc_auc_score(yval,probs)
print("AUC score : ",auc_score)
plt.plot(p_fpr, p_tpr, linestyle='--')
plt.plot(fpr, tpr, marker='.', label='Linear SVC (area=%0.3f)'% auc_score)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC-AUC CURVE for Linear SVC ")
plt.legend()
plt.show()
Output:
 
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
rbf_svm = SVC(kernel='rbf',probability=True)
splitter = StratifiedShuffleSplit(n_splits=10,random_state=101,test_size=0.2)
cv = cross_validate(estimator=rbf_svm,X=xtrain,y=ytrain,cv=splitter,n_jobs=-1,scoring='precision',
 return_estimator=True,return_train_score=True)
plt.plot(range(len(cv['test_score'])),cv['test_score'],label='test score',color='crimson')
plt.plot(range(len(cv['train_score'])),cv['train_score'],label='train score')
plt.scatter(range(len(cv['test_score'])),cv['test_score'],color='crimson')
plt.scatter(range(len(cv['train_score'])),cv['train_score'])
plt.xlabel('Model')
plt.ylabel('Precision')
plt.xticks(range(len(cv['test_score'])))
plt.legend()
plt.title('Cross Validation Results')
plt.show()
best_rbf_svc = cv['estimator'][1]
best_rbf_svc
params={'C':np.arange(1,5)}
tuner = RandomizedSearchCV(estimator=best_rbf_svc,param_distributions=params,n_jobs=-1,scoring='precision',
 cv=splitter,random_state=101,return_train_score=True,)
tuner.fit(xtrain,ytrain)
print("Hyper Parameter Tuning Results")
print("Best Params : ",tuner.best_params_)
print("Best Score : ",tuner.best_score_)
print("Best Model : ",tuner.best_estimator_)
best_rbf_svc = tuner.best_estimator_
ypred = best_rbf_svc.predict(xval)
ytrue = yval
print("CLASSIFICATION REPORT OF VALIDATION DATA")
print(classification_report(ytrue,ypred))
print("CONFUSION MATRIX OF VALIDATION DATA")
conf_mat = confusion_matrix(ytrue,ypred)
print(conf_mat)
print("Acuuracy : ",(conf_mat[0][0]+conf_mat[1][1])/len(ytrue))
probs = best_rbf_svc.predict_proba(xval)[:,1]
fpr,tpr,_ = roc_curve(yval,probs)
random_probs = [0 for i in range(len(yval))]
p_fpr,p_tpr,_ = roc_curve(yval,random_probs)
auc_score = roc_auc_score(yval,probs)
print("AUC score : ",auc_score)
plt.plot(p_fpr, p_tpr, linestyle='--')
plt.plot(fpr, tpr, marker='.', label='RBF SVC (area=%0.3f)'% auc_score)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC-AUC CURVE for RBF SVC ")
plt.legend()
plt.show()
 
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

