# Import essential libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score,f1_score,recall_score,confusion_matrix, classification_report

from sklearn.ensemble import RandomForestClassifier

# Read data
data=pd.read_csv(r"C:\Users\rinu_\Downloads\preprocessed_dataset.csv")


# Declare feature vector and target variable
y=data['Credit_Score']
X=data.drop('Credit_Score',axis=1)

# Split data into separate training and test set
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.25)

#Random Forest

rf_clf=RandomForestClassifier()
rf_clf.fit(X_train,y_train)
rf_pred=rf_clf.predict(X_test)

print('Accuracy:',accuracy_score(y_test,rf_pred))
print('Precision:',precision_score(y_test,rf_pred,average='weighted'))
print('recall:',recall_score(y_test,rf_pred,average='weighted'))
print('F1:',f1_score(y_test,rf_pred,average='weighted'))
print('classification_report:\n',classification_report(y_test,rf_pred))

print('\nConfusion Matrix:\n',confusion_matrix(y_test,rf_pred))
#create pickled file

import pickle
with open('model.pkl','wb') as model_file:
  pickle.dump(rf_clf,model_file)
