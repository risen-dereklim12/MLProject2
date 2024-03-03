import pandas as pd

data = pd.read_csv("data/cleaned_data.csv")

Y = data['Lung Cancer Occurrence'] #target variable
X = data.drop('Lung Cancer Occurrence', axis=1) #contains only the features
print (Y.head())
print (X.head())