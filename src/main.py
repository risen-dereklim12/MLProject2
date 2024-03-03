import pandas as pd
from sklearn.model_selection import train_test_split
from random_forest import RandomForest

data = pd.read_csv("data/cleaned_data.csv")

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

#creating an object of the RandomForest class
rf = RandomForest()
rf.fit(X_train, y_train, X_test, y_test)
rf.cross_val(X_train, y_train)
rf.confusion_matrix(X_test, y_test)
rf_predict = rf.predict(X_test, y_test)