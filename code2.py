# h1n1_vaccine_prediction
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
data_list = pd.read_csv("h1n1_vaccine_prediction.csv")
df = pd.DataFrame(data_list)
df.head()
df.shape
df.dtypes
df["age_bracket"] = df["age_bracket"].astype("category")
df["qualification"] = df["qualification"].astype("category")
df["race"] = df["race"].astype("category")
df["sex"] = df["sex"].astype("category")
df["income_level"] = df["income_level"].astype("category")
df["marital_status"] = df["marital_status"].astype("category")
df["housing_status"] = df["housing_status"].astype("category")
df["employment"] = df["employment"].astype("category")
df["census_msa"] = df["census_msa"].astype("category")
df.isnull()
df.isnull().sum()
df.info()
df.T.duplicated()
df = df.apply(lambda x: x.fillna(x.value_counts().index[0]))
df = pd.get_dummies(data=df)
df.isnull().sum()
y = df["h1n1_vaccine"]
x = df.drop("h1n1_vaccine", axis=1)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20, random_state=3)
pi_pr = RandomForestClassifier(n_estimators=100, random_state=10, max_depth=14,
                               oob_score=True, n_jobs=-1, criterion="gini", min_samples_split=5, max_features=6)
pi_pr.fit(x_train, y_train)
print(pi_pr.score(x_test, y_test))
y_prei_test = pi_pr.predict(x_test)
y_prei_test = pd.Series(y_prei_test)
print("The number test data are: \n(row, col)", x_test.shape)
print("\nPrediction of whether people got H1N1 vaccines for test data are : \n")
print(y_prei_test.value_counts())
y_prei_train = pi_pr.predict(x_train)
y_prei_train = pd.Series(y_prei_train)
print("The number train data are: \n(row, col)", x_train.shape)
print("\nPrediction of whether people got H1N1 vaccines for train data are : \n")
print(y_prei_train.value_counts())
