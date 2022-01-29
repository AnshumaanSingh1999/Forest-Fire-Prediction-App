import nltk
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
import math

t=int(sys.argv[1])
o=int(sys.argv[2])
h=int(sys.argv[3])

data=pd.read_csv("Forest_fire.csv")
data=np.array(data)


X=data[1:,1:-1]
y=data[1:,-1]
y=y.astype('int')
X=X.astype('int')
#print(X,y)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
log_reg=LogisticRegression()
lin_reg=LinearRegression()
svm_class = OneVsOneClassifier(SVC(random_state=0))
lin_reg.fit(X_train,y_train)
log_reg.fit(X_train,y_train)
svm_class.fit(X_train,y_train)
y_pred=lin_reg.predict(X_test)
b=log_reg.predict_proba(X_test)
inp=[[int(t),int(o),int(h)]]
a=log_reg.predict_proba(inp)
if a[0][1]>0.5:
    print("Probabilty of Forest Fire taking place is high.")
else:
    print("Probabilty of Forest Fire taking place is Low.")
