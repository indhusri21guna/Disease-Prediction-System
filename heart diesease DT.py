import pandas as pd #for reading dataset
import numpy as np # array handling functions

dataset = pd.read_csv("health_care.csv")#reading dataset
#print(dataset) # printing dataset

x = dataset.iloc[:,:-1].values #locating inputs
y = dataset.iloc[:,-1].values #locating outputs

#printing X and Y
print("x=",x)
print("y=",y)

from sklearn.model_selection import train_test_split # for splitting dataset
x_train,x_test,y_train,y_test = train_test_split(x ,y, test_size = 0.25 ,random_state = 0)
#printing the spliited dataset
print("x_train=",x_train)
print("x_test=",x_test)
print("y_train=",y_train)
print("y_test=",y_test)

from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier()

classifier.fit(x_train,y_train)#trainig Algorithm
y_pred=classifier.predict(x_test) #testing model
print("y_pred",y_pred) # predicted output
print("Testing Accuracy")
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

while True:

    a1=float(input("ENTER THE Glucose= "))
    b1=float(input("ENTER THE Cholesterol= "))
    c1=float(input("ENTER THE Hemoglobin= "))
    d1=float(input("ENTER THE Platelets= "))
    e1=float(input("ENTER THE White Blood Cells= "))
    f1=float(input("ENTER THE Red Blood Cells= "))
    g1=float(input("ENTER THE Hematocrit= "))
    h1=float(input("ENTER THE Mean Corpuscular Volume= "))
    i1=float(input("ENTER THE Mean Corpuscular Hemoglobin= "))
    j1=float(input("ENTER THE Mean Corpuscular Hemoglobin Concentration= "))
    k1=float(input("ENTER THE Insulin= "))
    l1=float(input("ENTER THE BMI= "))
    m1=float(input("ENTER THE Systolic Blood Pressure= "))
    n1=float(input("ENTER THE Diastolic Blood Pressure= "))
    o1=float(input("ENTER THE Triglycerides= "))
    p1=float(input("ENTER THE HbA1c= "))
    q1=float(input("ENTER THE LDL Cholesterol= "))
    r1=float(input("ENTER THE HDL Cholesterol= "))
    s1=float(input("ENTER THE ALT= "))
    t1=float(input("ENTER THE AST= "))
    u1=float(input("ENTER THE Heart Rate= "))
    v1=float(input("ENTER THE Creatinine= "))
    w1=float(input("ENTER THE Troponin= "))
    x1=float(input("ENTER THE C-reactive Protein= "))

    a=classifier.predict([[a1,b1,c1,d1,e1,f1,g1,h1,i1,j1,k1,l1,m1,n1,o1,p1,q1,r1,s1,t1,u1,v1,w1,x1]])
    print(a)
    if a=="Anemia":
        print("You're Anemia Patient")
    elif a=="Diabetes":
        print("You're Diabetic Patient")
    elif a=="Heart Di":
        print("You're Heart Patient")
    elif a=="Thalasse":
        print("You're Thalassemic Patient")
    elif a=="Thromboc":
        print("You're Thromboc Patient")
    elif a=="Healthy":
        print("You're Healthy Patient")
        
    print('Predicted new output value: %s' % (a))

            
