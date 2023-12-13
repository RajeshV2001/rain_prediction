import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
import seaborn as sns

from sklearn.model_selection import train_test_split

#loading Data
data=pd.read_csv('weather.csv')
df=pd.DataFrame(data)
df.dropna(axis=0,inplace=True)



#Removing RainToday Column
df.drop('RainToday',axis=1,inplace=True)

#Dependent and Independent Features
X=df.drop(['RainTomorrow'],axis=1)
y=df['RainTomorrow']
col=X.select_dtypes(include=['object'])

X.drop(col,axis=1,inplace=True)


lbl=LabelEncoder()
y=lbl.fit_transform(y)

X_train,X_test,y_train,y_test=train_test_split(X,y,shuffle=True,test_size=0.2,random_state=42,stratify=y)

scale=StandardScaler().fit(X_train)
X_train=scale.transform(X_train)
X_test=scale.transform(X_test)


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV,cross_val_score,KFold
from sklearn.metrics import mean_absolute_error,accuracy_score,classification_report,roc_curve

fold=KFold(n_splits=5)

clf=DecisionTreeClassifier(splitter='random',random_state=42)
model=clf.fit(X_train,y_train)

scores=cross_val_score(clf, X_train,y_train,scoring='neg_mean_squared_error',cv=fold)

pred=model.predict(X_test)

print("Mean Absolute Error -> ",mean_absolute_error(y_test,pred))
print("Train Score -> ",model.score(X_train,y_train))
print("Test Score  -> ",accuracy_score(y_test,pred))
print("Classifiaction Report :")
print(classification_report(y_test,pred))

