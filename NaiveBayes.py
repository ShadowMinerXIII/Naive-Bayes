from sklearn import preprocessing
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

Weather = ['Rain', 'Sunny', 'Murky', 'Rain', 'Murky', 'Sunny', 'Murky', 'Rain', 'Murky', 'Sunny']
Temperature = ['Cold', 'Hot', 'Cool', 'Cold', 'Cool', 'Hot', 'Cold', 'Cool', 'Hot', 'Hot']
Humidity = ['Low', 'High', 'Medium', 'Low', 'Medium', 'High', 'Low', 'High', 'Medium', 'High']
Wind_Speed = [30, 15, 40, 110, 26, 5, 19, 3, 39, 60]
Race = ['Yes', 'No', 'Yes', 'Yes', 'No', 'No', 'Yes', 'No', 'No', 'No']

le = preprocessing.LabelEncoder()
Weather_encoded=le.fit_transform(Weather)
Temperature_encoded=le.fit_transform(Temperature)
Humidity_encoded=le.fit_transform(Humidity)
label=le.fit_transform(Race)
features=list(zip(Weather_encoded, Temperature_encoded, Humidity_encoded, Wind_Speed))
clf = GaussianNB()
clf.fit(features, label)
GaussianNB(priors=None)
#print(clf.predict_proba(features))
#print(features)
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy Classification Score:",accuracy_score(y_test, y_pred, normalize=False))
#print("Precision Score:",metrics.precision_score(y_test, y_pred))
