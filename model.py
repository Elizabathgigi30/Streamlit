import pandas as pd
df=pd.read_csv('iris.csv')
print(df.head())
x=df[ ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'] ]
y=df[ ['Species'] ]
print(x.columns)
print (y.columns)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)

from sklearn.preprocessing import MinMaxScaler

min_max=MinMaxScaler()
x_train=min_max.fit_transform(x_train)
x_test=min_max.fit_transform(x_test)

from sklearn.tree import DecisionTreeClassifier
dc=DecisionTreeClassifier()
dc.fit(x_train,y_train)
y_pred=dc.predict(x_test)


#scaling to fit using minmax scaler


from sklearn.metrics import accuracy_score
print("Accuracy on Decision Tree Model: ", accuracy_score(y_test, y_pred))

import joblib

joblib.dump(dc,'dc_model')