# texi_fare_prediction
import pandas as pd
import numpy as np
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from geopy.distance import geodesic
import math
list_data = pd.read_csv("TaxiFare.csv")
df = pd.DataFrame(list_data)
df['unique_id'] = df['unique_id'].astype('category')
df['date_time_of_pickup'] = df['date_time_of_pickup'].astype('category')
df = df.drop("unique_id", axis=1)
df = df.drop("date_time_of_pickup", axis=1)
df.head()
df.shape
df.info()
df.isnull().sum()
df.dtypes
df.describe().T

sns.histplot(data=df, x=df["amount"])
sns.heatmap(df.corr(), annot=True)


def get_distance_km(latt1, long1, latt2, long2):
    if ((latt1 > -90 and latt1 < 90) and (latt2 > -90 and latt2 < 90)):
        p_point = (latt1, long1)
        d_point = (latt2, long2)
        distance_km = geodesic(p_point, d_point).km
        return distance_km


df_len = len(df)
distances_km = []
for i in range(len(df)):
    distance = get_distance_km(df.iloc[i, df.columns.get_loc('latitude_of_pickup')], df.iloc[i, df.columns.get_loc(
        'longitude_of_pickup')], df.iloc[i, df.columns.get_loc('latitude_of_dropoff')], df.iloc[i, df.columns.get_loc('longitude_of_dropoff')])
    distances_km.append(distance)
df["distance"] = distances_km

y = df["amount"]
x = df.drop("amount", axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=3)
df_train = xgb.DMatrix(x_train, label=y_train)
df_test = xgb.DMatrix(x_test, label=y_test)
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': 5,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'verbosity': 0,
    'seed': 3
}
num_round = 100
model = xgb.train(params, df_train, num_round)

prdi = model.predict(df_test)
rmse = math.sqrt(mean_squared_error(y_test, prdi))
print("----", rmse, "----")
