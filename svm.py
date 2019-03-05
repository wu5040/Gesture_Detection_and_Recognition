import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

def getDataFromCSV():
    f = open("trainData.csv", 'r')
    csvreader = csv.reader(f)
    trainData = list(csvreader)

    f = open("trainLabels.csv", 'r')
    csvreader = csv.reader(f)
    trainLabels= list(csvreader)

    return trainData,trainLabels[0]

Data,Labels=getDataFromCSV()

Data=np.asarray(Data)
Labels=np.asarray(Labels)

X_train, X_test, y_train, y_test = train_test_split(Data, Labels, test_size=0.33, random_state=42)

classif = OneVsRestClassifier(SVC(C=2,kernel='rbf',gamma="scale"))
classif.fit(X_train, y_train)

pre_results=classif.predict(X_test)

right=0
for i in range(len(y_test)):
    if pre_results[i]==y_test[i]:
        right+=1

print(right/len(y_test)*100)