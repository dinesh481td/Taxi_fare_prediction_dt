#import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.pipeline import Pipeline
from feature_engine.outliers import Winsorizer
import sidetable as stb
from sqlalchemy import create_engine
import mysql.connector as connector
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler
import sweetviz
import joblib 
import pickle
from sklearn.impute import SimpleImputer
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


#Load the dataset
data1 = pd.read_csv(r"C:/Users/Dinesh T/Downloads/yellow_tripdata_2020-01/yellow_tripdata_2020-01.csv")


'''
engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                       .format(user = "root",# user
                               pw = "user1", # passwrd
                               db = "Taxi")) #database

data.to_sql('Taxi', con = engine, if_exists = 'replace', chunksize = 6500000, index = False)


#pip install mysql
#pip install mysql-connector-python
import mysql.connector as connector

con = connector.connect(host = 'localhost',
                  port = '3306',
                  user = 'root',
                  password = 'user1',
                  database = 'Taxi',
                  auth_plugin = 'mysql_native_password')

cur = con.cursor()
con.commit()

cur.execute('SELECT * FROM taxi')
df = cur.fetchall()

data1 = pd.DataFrame(df)
data1.shape
data1.columns

data1 = data1.rename({0:'VendorID', 1:'tpep_pickup_datetime', 2:'tpep_dropoff_datetime', 3:'tpep_dropoff_datetime',
                      4:'trip_distance', 5:'RatecodeID', 6:'store_and_fwd_flag', 7:'PULocationID', 8: 'DOLocationID',
                      9:'payment_type', 10:'fare_amount', 11: 'extra', 12: 'mta_tax', 13:'tip_amount', 14: 'tolls_amount',
                      15: 'improvement_surcharge', 16: 'total_amount', 17:'congestion_surcharge'}, axis=1)

#Autoeda
data1.info()
data1.describe()
report = sweetviz.analyze([data1, "data"])
report.show_html('Report1.html')
'''



# Preprocessing

data1.columns
#Type casting
data1["tpep_pickup_datetime"] = data1["tpep_pickup_datetime"].astype("str")
data1["tpep_pickup_datetime"] = pd.to_datetime(data1["tpep_pickup_datetime"])

data1["tpep_dropoff_datetime"] = data1["tpep_dropoff_datetime"].astype("str")
data1["tpep_dropoff_datetime"] = pd.to_datetime(data1["tpep_dropoff_datetime"])

data1.dtypes

# Handling Duplicates 
data1.duplicated().sum()

#There are 12949 duplicates values found.Hence, dropping them
#Duplicates percentage 0.2%
# Removing Duplicated Values
df = data1.drop_duplicates()

# Missing Value
# There are 0.8% missing values along with duplicated values in the dataset
data1.isnull().sum()
(data1.isnull().sum() /len(df)).sum()

#df = data1.dropna()



# Again check for missing values 4.0%
df.isnull().sum()

# Zero variance 
df.var()

df_flag = df['store_and_fwd_flag'].value_counts()/len(df)

fig1, ax1 = plt.subplots(figsize=(8, 8))
ax1.pie(df_flag, labels=df_flag.index, autopct='%1.2f%%')
ax1.legend(title='store_and_fwd_flag_count', loc='upper right', bbox_to_anchor=(1.0, 1.0), labels=["{}, {:.2f}%".format(l, s) for l, s in zip(df_flag.index, df_flag)])
plt.title('store_and_fwd_flag')
plt.show()


# No = 0.989255, Yes = 0.010745 The majority of entries are same so we remove the column.
df1 = df.drop(["store_and_fwd_flag"],1)

#Calculating time in minutes by using trip pickup time and drop off time
df1.info()
df1["time_difference"] =  df1["tpep_dropoff_datetime"] - df1["tpep_pickup_datetime"]
df1["time_difference_minutes"] = [(i.total_seconds()/60) for i in df1["time_difference"]]

df1 = df1.drop(["tpep_dropoff_datetime","tpep_pickup_datetime"],1)
df1 = df1.drop(["time_difference"],1)

df1.columns

from statsmodels.stats.outliers_influence import variance_inflation_factor
#vif = [variance_inflation_factor(df1.values,i) for i in list(range(df1.shape[1]))]

######################
# Unique values of passenger count
df1["passenger_count"].unique()
df1["passenger_count"].value_counts()/len(df1)

Group_pass = df1.loc[df1['passenger_count'] == 0.0]
# sns.heatmap(Group_pass.corr(),annot = True)
# sns.pairplot(Group_pass)
df2 = df1[df1['passenger_count'] != 0.0]

df2.isnull().sum()
df3 = df2[df2['RatecodeID'] != 99]

#Making new feature New_total_amount by adding features like total_amount and congestion_surcharge
df3["New_total_amount"] = df3["total_amount"] + df3["congestion_surcharge"]

df3["trip_distance(m)"] = df3["trip_distance"]*1609.34

df3 = df3.drop(["VendorID","fare_amount", "congestion_surcharge","extra", "mta_tax", "tip_amount",
                "tolls_amount","improvement_surcharge","total_amount", "trip_distance"],1)
df3.columns


df4 = df3

num_features1 = ['passenger_count', 'RatecodeID', 'PULocationID', 'DOLocationID','payment_type']
num_features1

num_features2 = ['time_difference_minutes', 'trip_distance(m)']
num_features2
df4.isna().sum()
####################### Pipeline for missing values for integer (categorical data)
from sklearn.impute import SimpleImputer

num_pipeline1 = Pipeline(steps = [('impute1', SimpleImputer(strategy='most_frequent'))])


#pipeline for missing values for float (continuous data)

num_pipeline2 = Pipeline(steps = [('impute2', SimpleImputer(strategy='mean'))])

preprocessor = ColumnTransformer([('most_frequent', num_pipeline1, num_features1),
                                  ('median', num_pipeline2, num_features2)],remainder="passthrough")

imputation = preprocessor.fit(df4)

#save the impute model
joblib.dump(imputation, 'impute')

#Transform the original data
num_data = pd.DataFrame(imputation.transform(df4), columns = df4.columns)

df4.payment_type.value_counts()
num_data.payment_type.value_counts()
num_data.dtypes
num_data.isnull().sum()

num_data.duplicated().sum()
#############################
df4.isnull().sum()
df4.info()

df5 = df4.iloc[:,[0,1,2,3,4,5,7,6]]
df5.isna().sum()
df5 = df5.dropna()
df5.duplicated().sum()
df5 = df5.drop_duplicates()



#####################################

#Winsorization

#Outlier Analysis
df5.plot(kind = 'box', subplots = True, sharey = False, figsize = (10, 6)) 
# increase spacing between subplots
plt.subplots_adjust(wspace = 1)
plt.show()

#Winsorization for 'trip distance' & 'time difference in minutes' columns

winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=["time_difference_minutes","trip_distance(m)",'New_total_amount'])

clean = winsor.fit(df5[["time_difference_minutes","trip_distance(m)",'New_total_amount']])

#save the winsor model
joblib.dump(clean, 'winsor')

df5[['time_difference_minutes','trip_distance(m)','New_total_amount']] = clean.fit_transform(df5[['time_difference_minutes','trip_distance(m)','New_total_amount']])
df5[['time_difference_minutes', 'trip_distance(m)','New_total_amount']] = df5[['time_difference_minutes','trip_distance(m)','New_total_amount']].astype('float64')
#Again check for outliers
df5.plot(kind = 'box', subplots = True, sharey = False, figsize = (10, 6)) 
# increase spacing between subplots
plt.subplots_adjust(wspace = 1)
plt.show()

df5.duplicated().sum()
df5 = df5.drop_duplicates()
df5.info()
'''
By adding these two columns together, we get a combined measure of the length of the trip 
that takes into account both the physical distance traveled and the time it took to travel that distance. 
This combined measure is saved in a new column called "trip_length".

'''

from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import r2_score
#Train test split
df6 = df5
df6.info()
df6.duplicated().sum()
df6 = df6.drop_duplicates()

x_new = df6.drop(["New_total_amount"],1)
vif1 = [variance_inflation_factor(x_new.values,i) for i in list(range(x_new.shape[1]))]
correlation1 = df6.drop(["PULocationID","DOLocationID"],1)
correlation1.corr()
#x_new = sm.add_constant(x_new)
y_new = df6.New_total_amount
df6.isna().sum()

X_train, X_test, Y_train, Y_test = train_test_split(x_new,y_new,test_size = 0.3, random_state = 0)

# draft model
#p = sm.add_constant(df5.iloc[:,5,6])
base = sm.OLS(df5.iloc[:,-1],df5.iloc[:,:7]).fit()
base.summary()

######################################## Model 1

base_1 = sm.OLS(Y_train,X_train).fit()
base_1.summary()
df6.corr()
base1_pred_train = base_1.predict(X_train)
r2_score(Y_train,base1_pred_train)
base1_pred_test = base_1.predict(X_test)
r2_score(Y_test,base1_pred_test)
# After doing Winsorization our accuracy for train = 0.84 and test = 0.84
 
######################################## Model 2 -Without winsorization
winsored_df = df4.dropna()
winsored_df.isnull().sum()
winsored_df.duplicated().sum()
winsored_df = winsored_df.drop_duplicates()

winsored_df.plot(kind = 'box', subplots = True, sharey = False, figsize = (10, 6)) 
x_new_1 = winsored_df.drop(["New_total_amount"],1)
y_new_1 = winsored_df["New_total_amount"]
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(x_new_1,y_new_1,test_size = 0.2, random_state = 0)

base_2 = sm.OLS(Y_train1,X_train1).fit()
base_2.summary()

base2_pred_train = base_2.predict(X_train1)
r2_score(Y_train1,base2_pred_train)
base2_pred_test = base_2.predict(X_test1)
r2_score(Y_test1,base2_pred_test)
# Without winsorization we get train = 0.79 , tes = 0.80

######################################## Model 3

from sklearn.linear_model import Lasso,Ridge,ElasticNet
from sklearn.model_selection import GridSearchCV
params = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}
grid1 = GridSearchCV(Lasso(), params,scoring="r2",cv = 5)
grid_model1 = grid1.fit(X_train,Y_train)
lasso_model = grid_model1.best_estimator_

lasso_pred_train = lasso_model.predict(X_train)
r2_score(Y_train,lasso_pred_train)
lasso_pred_test = lasso_model.predict(X_test)
r2_score(Y_test,lasso_pred_test)
# After doing Winsorization our accuracy for train = 0.88 and test = 0.88

######################################## Model 4

grid2 = GridSearchCV(Ridge(), params,scoring="r2",cv = 5)
grid_model2 = grid2.fit(X_train,Y_train)
ridge_model = grid_model2.best_estimator_

ridge_pred_train = ridge_model.predict(X_train)
r2_score(Y_train,ridge_pred_train)
ridge_pred_test = ridge_model.predict(X_test)
r2_score(Y_test,ridge_pred_test)
# After doing Winsorization our accuracy for train = 0.88 and test = 0.88
######################################## model 5

grid3 = GridSearchCV(ElasticNet(), params,scoring="r2",cv = 5)
grid_model3 = grid3.fit(X_train,Y_train)
ElasticNet_model = grid_model3.best_estimator_

elastic_pred_train = ElasticNet_model.predict(X_train)
r2_score(Y_train,elastic_pred_train)
elastic_pred_test = ElasticNet_model.predict(X_test)
r2_score(Y_test,elastic_pred_test)
# After doing Winsorization our accuracy for train = 0.88 and test = 0.88
######################################### Model 5 --BEST MODEL #######################################

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
dt = DecisionTreeRegressor(min_samples_split=60,min_samples_leaf=3)
dt_model = dt.fit(X_train,Y_train)
dt_pred_train = dt_model.predict(X_train)
r2_score(Y_train,dt_pred_train)
dt_pred_test = dt_model.predict(X_test)
r2_score(Y_test,dt_pred_test)
rmse_dt_train = rmse(Y_train,dt_pred_train)
mae_dt_train = mae(Y_train, dt_pred_train)
rmse_dt_test = rmse(Y_test,dt_pred_test)
mae_dt_test = mae(Y_test, dt_pred_test)
######################################## 