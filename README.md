# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required library packages.
2. Import the dataset to operate on.
3. Split the dataset into required segments.
4. Predict the required output.
5. Run the program

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: MOHAMED ROSHAN S
RegisterNumber:  212222040101
*/
```
```py
import chardet
file ='spam.csv'
with open(file, 'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding='Windows-1252')

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

```

## Output:
## RESULT OUTPUT
![SVM For Spam Mail Detection](o1.png)
## DATA HEAD()
![alt](o2.png)
## DATA INFO()
![alt](o3.png)
## DATA ISNULL().SUM()
![alt](o4.png)
## Y_PREDICTION VALUE
![ALT](o5.png)
## ACCURACY VALUE
![ALT](o6.png)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
