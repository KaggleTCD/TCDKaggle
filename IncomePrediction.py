import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


#Target encoding function
def calc_target_encoding(df, by, on, m):
    mean = df[on].mean()
    agg = df.groupby(by)[on].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']
    
    smooth = (counts * means + m * mean) / (counts + m)
    
    return df[by].map(smooth)

#Import datasets
data_train = pd.read_csv('F:/MS/Masters/Machine Learning/TrainData.csv')
data_test = pd.read_csv('F:/MS/Masters/Machine Learning/TestData.csv')
#Renaming Income in EUR to Income to shorten it
data_train  = data_train.rename(index=str, columns={"Income in EUR" : "Income"})

#Conactenating train and test datasets so that columns after one hot
#encoding match in case there are some different values in the same
#column of train and test dataset.
data = pd.concat([data_train, data_test], sort=False)

#Renaming the column names to one word
data = data.drop("Instance", axis=1)     
data = data.rename(index=str, columns={"Body Height [cm]" : "Height"})
data = data.rename(index=str, columns={"Year of Record" : "YearOfRecord"})
data = data.rename(index=str, columns={"Size of City" : "SizeOfCity"})
data = data.rename(index=str, columns={"University Degree": "UniversityDegree"})
data = data.rename(index=str, columns={"Wears Glasses" : "WearsGlasses"})
data = data.rename(index=str, columns={"Hair Color" : "HairColor"})
data = data.rename(index=str, columns={"Income in EUR" : "Income"})

data['Gender'] = data['Gender'].replace('0', "other") 
data['Gender'] = data['Gender'].replace('unknown', pd.np.nan) 

#Label encoding the university degree
data['UniversityDegree'] = data['UniversityDegree'].replace('PhD', 4) 
data['UniversityDegree'] = data['UniversityDegree'].replace('Master', 3) 
data['UniversityDegree'] = data['UniversityDegree'].replace('Bachelor', 2) 
data['UniversityDegree'] = data['UniversityDegree'].replace('No', 0) 
data['UniversityDegree'] = data['UniversityDegree'].replace(pd.np.nan, 0) 

data['HairColor'] = data['HairColor'].replace('0', pd.np.nan) 
data['HairColor'] = data['HairColor'].replace('Unknown', pd.np.nan)




data['Country'] = calc_target_encoding(data, 'Country', 'Income', 2)
data['Profession'] = calc_target_encoding(data, 'Profession', 'Income', 50)


data.drop("YearOfRecord", axis=1)
data.drop("Country", axis=1)
data.drop("WearsGlasses", axis=1)

data1 = pd.get_dummies(data, columns=["Gender"], drop_first = True)

data1 = pd.get_dummies(data1, columns=["HairColor"], drop_first = True)

pd.set_option('display.max_columns', 100)

X_train = data1[0:len(data_train)]


X_train["YearOfRecord"].fillna((X_train["YearOfRecord"].mean()), inplace=True )
X_train["Age"].fillna((X_train["Age"].mean()), inplace=True )
X_train["Profession"].fillna((X_train["Profession"].mean()), inplace=True )

X_train.isnull().sum()
Y_train = X_train[["Income"]]

X_train = X_train.drop("Income", axis=1)

X_trainData, X_testData, Y_trainData, Y_testData = train_test_split(X_train, Y_train, train_size=0.8, random_state=100)

randomForestRegressor = RandomForestRegressor(n_estimators= 900, random_state=100)
print("Done till here")
model = randomForestRegressor.fit(X_trainData, Y_trainData)
print("Before prediction")
Ypred = model.predict(X_testData)
print("After prediction")
import math
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y_testData, Ypred)
rmse = math.sqrt(mse)
rmse
print(rmse)
print("Getting TestData")
X_test = data1[len(data_train):]
print("TestData ",X_test)
X_test = X_test.drop("Income", axis=1)
print("X_test after dropping Income ",X_test)
X_test.head()

print(X_test.isnull().sum())
print()
X_test["YearOfRecord"].fillna((X_test["YearOfRecord"].mean()), inplace=True )
X_test["Age"].fillna((X_test["Age"].mean()), inplace=True )
X_test["Profession"].fillna((X_test["Profession"].mean()), inplace=True )
X_test["Country"].fillna((X_test["Country"].mean()), inplace=True )
print(X_test.isnull().sum())
Y_pred = model.predict(X_test)
Y_pred = pd.DataFrame(Y_pred)

Y_pred.to_csv("F:/MS/Masters/Machine Learning/Y_pred_submission.csv")