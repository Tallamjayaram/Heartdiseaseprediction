import numpy as np
import pandas as pd
data=pd.read_csv("heart.csv")
data.head()
data.isnull().sum()
data['output']
X=data.drop('output',axis=1)
Y=data['output']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=1)
from sklearn.linear_model import LogisticRegression
l=LogisticRegression()
l.fit(X_train,y_train)
ypred=l.predict(X_test)
ypred
from sklearn.metrics import classification_report,accuracy_score
accuracy_score(ypred,y_test)*100
print(classification_report(ypred,y_test))
print(X)
from flask import Flask,request,jsonify
app=Flask("Heart Disease Prediction")
@app.route('/hello')
def new():
    return "hey hello how are you"
@app.route("/<int:age>/<int:sex>/<int:cp>/<int:trtbps>/<int:chol>/<int:fbs>/<int:restecg>/<int:thalachh>/<int:exng>/<float:oldpeak>/<int:slp>/<int:caa>/<int:thall>")

def test(age,sex,cp,trtbps,chol,fbs,restecg,thalachh,exng,oldpeak,slp,caa,thall):
    p=[]
    p+=[age,sex,cp,trtbps,chol,fbs,restecg,thalachh,exng,oldpeak,slp,caa,thall]
    #CONVERT THE DATA INTO ARRAY
    arr=np.array([p])
    predict=l.predict(arr)
    if predict==[1]:
        result={'result':'High chance of Heart Disease'}
    else:
        result={"result":"Low chance of Heart Disease"}
    return result      
app.run()
