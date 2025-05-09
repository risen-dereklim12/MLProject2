import pandas as pd
from sklearn.model_selection import train_test_split
from decision_tree import DecisionTree
from random_forest import RandomForest
from logistic_regression import LogRegression

data = pd.read_csv("cleaned_data.csv")

Y = data['Lung Cancer Occurrence'] #target variable
X = data.drop('Lung Cancer Occurrence', axis=1) #contains only the features
# Drop the old columns without one-hot encoding and the columns with range of values 
# for visualizing the distribution
X.drop(['Gender', 'COPD History', 'Genetic Markers', 'Air Pollution Exposure', 'Start Smoking',
        'Stop Smoking', 'Taken Bronchodilators', 'Frequency of Tiredness', 'Dominant Hand',
        'Age Group', 'Weight Change Group', 'Years Quitted Smoking Group']
        , axis=1, inplace=True)

#train test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

print ("Shape of X_train = ",X_train.shape)
print ("Shape of y_train = ",y_train.shape)
print ("Shape of X_test = ",X_test.shape)
print ("Shape of Y_test = ",y_test.shape)

def decision_tree():
    #creating an object of the Decision Tree class
    dt = DecisionTree()
    dt.fit(X_train, y_train, X_test, y_test)
    dt.cross_val(X_train, y_train)
    dt.confusion_matrix(X_test, y_test)
    dt.predict(X_test, y_test)

def random_forest():
    #creating an object of the RandomForest class
    rf = RandomForest()
    rf.fit(X_train, y_train, X_test, y_test)
    rf.predict(X_train, y_train, X_test, y_test)

def log_regression():
    #creating an object of the LogRegression class
    clf = LogRegression()
    clf.fit(X_train, y_train, X_test, y_test)

input = input("Enter 1 for Decision Tree, 2 for Random Forest, 3 for Logistic Regression: ")
if input == '1':
     decision_tree()
elif input == '2':
     random_forest()
elif input == '3':
     log_regression()

