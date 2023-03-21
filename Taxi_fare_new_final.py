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
from sklearn.preprocessing import MinMaxScaler
import sweetviz
import joblib 
import pickle
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

#Load the dataset
data1 = pd.read_csv(r"C:/Users/Dinesh T/Downloads/yellow_tripdata_2020-01/yellow_tripdata_2020-01.csv")

#Pusing the dataset to sql
engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                       .format(user = "root",# user
                               pw = "user1", # passwrd
                               db = "Taxi")) #database

data.to_sql('taxi', con = engine, if_exists = 'replace', chunksize = 6500000, index = False)

#Pulling the dataset back to python
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
data.columns

#Renaming the column after pulling the dataset from sql
data1 = data1.rename({0:'VendorID', 1:'tpep_pickup_datetime', 2:'tpep_dropoff_datetime', 3:'passenger_count',
                      4:'trip_distance', 5:'RatecodeID', 6:'store_and_fwd_flag', 7:'PULocationID', 8: 'DOLocationID',
                      9:'payment_type', 10:'fare_amount', 11: 'extra', 12: 'mta_tax', 13:'tip_amount', 14: 'tolls_amount',
                      15: 'improvement_surcharge', 16: 'total_amount', 17:'congestion_surcharge'}, axis=1)

#Autoeda
data1.info()
data1.describe()
report = sweetviz.analyze([data1, "data"])
report.show_html('Report1.html')

#preprocessing
# Handling Duplicates 
data1.duplicated().sum()

#There are 12949 duplicates values found.Hence, dropping them
#Duplicates percentage 0.2%
# Removing Duplicated Values
df = data1.drop_duplicates()
df.dtypes

#Type casting(converting object to date and time)
df["tpep_pickup_datetime"] = df["tpep_pickup_datetime"].astype("str")
df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])

df["tpep_dropoff_datetime"] = df["tpep_dropoff_datetime"].astype("str")
df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])

df.dtypes

# Missing Value
#There are 0.8% missing values in the dataset
df.isnull().sum()

df = df.dropna()
#Again check for missing values
df.isnull().sum()
#No missing values

# Zero variance 
df.var()

df_flag = df['store_and_fwd_flag'].value_counts()

fig1, ax1 = plt.subplots(figsize=(8, 8))
ax1.pie(df_flag, labels=df_flag.index, autopct='%1.2f%%')
ax1.legend(title='store_and_fwd_flag_count', loc='upper right', bbox_to_anchor=(1.0, 1.0), labels=["{}, {:.0f}".format(l, s) for l, s in zip(df_flag.index, df_flag)])
plt.title('store_and_fwd_flag')
plt.show()

# No = 0.989255, Yes = 0.010745 The majority of entries are same so we remove the column.
df1 = df.drop(["store_and_fwd_flag"],1)

#Calculating time in minutes by using trip pickup time and drop off time

df1["time_difference"] =  df1["tpep_dropoff_datetime"] - df1["tpep_pickup_datetime"]
df1["time_difference_minutes"] = [(i.total_seconds()/60) for i in df1["time_difference"]]

df1 = df1.drop(["tpep_dropoff_datetime","tpep_pickup_datetime"],1)
df1 = df1.drop(["time_difference"],1)
df1.columns

########################EDA################################

# First momment bussiness decision
df1.mean()
df1.median()

# Second momment bussiness decision
df1.var() 
df1.std()

# Third momment bussiness decision
df1.skew() 


# Fourth momment bussiness decision
df1.kurt()

#Histogram
'''
num_df1 = df1.select_dtypes(exclude ="object" )
for i, col in enumerate(num_df1.columns):
    fig, ax = plt.subplots()
    ax.set_xlabel(col, rotation=0, fontsize=12)
    df1[col].plot(kind="hist", ax=ax,bins = 20)
    plt.show()
'''

#Unique values of vendors
df1["VendorID"].unique()
Pie_VendorID = df1['VendorID'].value_counts()
plt.pie(Pie_VendorID, labels=Pie_VendorID.index, autopct='%1.2f%%')
plt.title('VendorID')
plt.show()
# vendorID = 1= Creative Mobile Technologies, LLC; 2= VeriFone Inc. (Taxicab Passenger Enhancement Program (TPEP) to record trip data.)
# TPEP Provider VeriFone Inc has give majority of Records 66.96%
# The rest provider is Creative Mobile Technologies 33.04%

#Unique values of passenger count
df1["passenger_count"].unique()
df1["passenger_count"].value_counts()/len(df1)

Group_pass = df1.loc[df1['passenger_count'] == 0.0]

df2 = df1[df1['passenger_count'] != 0.0]
#1.8% of records got dropped

Pie_passenger_count = (df2['passenger_count'].value_counts()/len(df2)*100)

fig1, ax1 = plt.subplots(figsize=(8, 15))
ax1.pie(Pie_passenger_count)
ax1.legend(title='passenger_count', loc='upper right', bbox_to_anchor=(1.0, 1.0), labels=["{}, {:.2f}%".format(l, s) for l, s in zip(Pie_passenger_count.index, Pie_passenger_count)])
plt.title('passenger_count')
plt.show()

# Passenger_count column contains a value of 0.0 and it is invalid if 0 passengers have booked the taxi.So we dropping those columns.
# In passenger_count the propostion of single Customer booking a cab is 73.04%.

vis_dist = pd.cut(df1["trip_distance"], 
             bins = [min(df1.trip_distance),
                     df1.trip_distance.quantile(0.25),
                     df1.trip_distance.quantile(0.75),
                     max(df1.trip_distance)],
                     labels=["low distance","Average Distance", "Maximum Distance"])

vis_dist = pd.DataFrame(vis_dist)

freq_vis_dist = vis_dist.groupby(["trip_distance"]).size().reset_index(name='Count')
freq_vis_dist
fig1, ax1 = plt.subplots(figsize=(10, 8))
sns.barplot(x="trip_distance",y="Count", data=freq_vis_dist)

# Add value labels
ax = plt.gca()
for p in ax.patches:
    ax.text(p.get_x() + p.get_width() / 2., p.get_height(), '%.0f' % p.get_height(), 
            fontsize=12, color='black', ha='center', va='bottom')

# Add a title
plt.title('trip_distance')

#trip distance unique count
dis = df1["trip_distance"].value_counts()

# trip_distance contains 0.0 as distance value in 67476 rows and it is invalid value for distance we replace it with mean of trip_distance column.
#RatecodeID unique count
df2["RatecodeID"].value_counts()
#RatecodeID column contains 99, which is invalid and dropping those records
Group_rate = df2.loc[df2['RatecodeID'] > 6]
df3 = df2[df2['RatecodeID'] != 99]
#0.004% of records got dropped
Pie_Rate = df3["RatecodeID"].value_counts()
df3["RatecodeID"].value_counts() / len(df3)

#Dropping the records where rate id code columns having value 99
sns.histplot(df3["RatecodeID"],bins = 5)
plt.title('RatecodeID')
plt.show()
# "RatecodeID" indicate that 96.65% of customers booked the taxi for 1 = Standard rate.
data.columns

#Unique counts of paymenttype
Pie_paymenttype = (df3["payment_type"].value_counts()/len(df3))*100
Pie_paymenttype.index
df3["payment_type"] = df3["payment_type"].astype('int64')
# payment_type Credit card and Cash are used most often by customers.

sns.histplot(df3["payment_type"],bins=10)
plt.title('paymenttype')
plt.show()

#pickup location
df3["PULocationID"].value_counts() 
df3.stb.freq(["PULocationID"])

#Top 20 pickup location
plt.figure(figsize=(20,10))
df3['PULocationID'].value_counts().nlargest(20).plot(kind='bar')
plt.title('Top 20 pickup locations')
plt.show()

'''
#For sample we take two pick location and their details
    According to TLC's documentation, PULocationID 237 corresponds to the "Upper East Side North" taxi zone, 
    which includes the area bounded by East 96th Street to the north, 
    Central Park to the west, East 59th Street to the south, and the East River to the east.
    
    161 refers to the "Crown Heights North" taxi zone.
    This zone is located in Brooklyn and is roughly bounded by Atlantic Avenue to the north, 
    Utica Avenue to the east, Eastern Parkway to the south, and Washington Avenue to the west. 
    It includes several notable landmarks such as the Brooklyn Museum, the Brooklyn Botanic Garden,
    and the Brooklyn Children's Museum.
'''

df3["DOLocationID"].value_counts()
df3.stb.freq(["DOLocationID"])


#Top 20 dropoff location
plt.figure(figsize=(20,10))
df3['DOLocationID'].value_counts().nlargest(20).plot(kind='bar')
plt.title('Top 20 dropoff locations')
plt.show()

'''
#For sample we take the top 2 drop off locations and their details
# DOLocationID 237 and 236 was booked by customer most often.   
    236 refers to the "Upper East Side South" taxi zone. 
    This zone is located in Manhattan and is roughly bounded by East 59th Street to the north, 
    Fifth Avenue to the west, East 42nd Street to the south, and the East River to the east. 
    It includes several notable landmarks such as the Plaza Hotel, Bloomingdale's, and Central Park.
'''   
#Timedifference
df3['time_difference_minutes'].value_counts()

#Fare amount
df3['fare_amount'].value_counts().shape


vis_fare_amt = pd.cut(df3["fare_amount"], 
             bins = [min(df3.fare_amount),
                     df3.fare_amount.quantile(0.25),
                     df3.fare_amount.quantile(0.75),
                     max(df3.fare_amount)],
                     labels=["low_Fare_amount ","Average_Fare_amount", "High_Fare_amount"])

vis_fare_amt = pd.DataFrame(vis_fare_amt)

freq_vis_fare_amt = vis_fare_amt.groupby(["fare_amount"]).size().reset_index(name='sum')
freq_vis_fare_amt

sns.barplot( x="fare_amount",y="sum", data=freq_vis_fare_amt)

# Add value labels
ax = plt.gca()
for p in ax.patches:
    ax.text(p.get_x() + p.get_width() / 2., p.get_height(), '%.0f' % p.get_height(), 
            fontsize=12, color='black', ha='center', va='bottom')
############################################################

#sns.histplot(df3['fare_amount'])
#Making new feature New_total_amount by adding features like total_amount and congestion_surcharge
df3["New_total_amount"] = df3["total_amount"] + df3["congestion_surcharge"]

df3["trip_distance(m)"] = df3["trip_distance"]*1609.34
df3.duplicated().sum()
df3 = df3.drop(["VendorID","fare_amount", "congestion_surcharge","extra", "mta_tax", "tip_amount",
               "tolls_amount","improvement_surcharge","total_amount", "trip_distance"],1)
df3.columns
df3.duplicated().sum()

######################################################
 

vis_total_amt = pd.cut(df3["New_total_amount"], 
             bins = [min(df3.New_total_amount),
                     df3.New_total_amount.quantile(0.25),
                     df3.New_total_amount.quantile(0.75),
                     max(df3.New_total_amount)],
                     labels=["low_total_amount ","Average_total_amount", "High_total_amount"])

vis_total_amt = pd.DataFrame(vis_total_amt)

freq_vis_total_amt = vis_total_amt.groupby(["New_total_amount"]).size().reset_index(name='sum')
freq_vis_total_amt

sns.barplot( x="New_total_amount",y="sum", data=freq_vis_total_amt)

# Add value labels
ax = plt.gca()
for p in ax.patches:
    ax.text(p.get_x() + p.get_width() / 2., p.get_height(), '%.0f' % p.get_height(), 
            fontsize=12, color='black', ha='center', va='bottom')

#####################################################



vis_mins_amt = pd.cut(df3["time_difference_minutes"], 
             bins = [min(df3.time_difference_minutes),
                     df3.time_difference_minutes.quantile(0.25),
                     df3.time_difference_minutes.quantile(0.75),
                     max(df3.time_difference_minutes)],
                     labels=["low_mins_diff","Average_mins_diff", "High_mins_diff"])

vis_mins_amt = pd.DataFrame(vis_mins_amt)

freq_vis_mins = vis_mins_amt.groupby(["time_difference_minutes"]).size().reset_index(name='mean')
freq_vis_mins

sns.barplot( x="time_difference_minutes",y="mean", data=freq_vis_mins)

# Add value labels
ax = plt.gca()
for p in ax.patches:
    ax.text(p.get_x() + p.get_width() / 2., p.get_height(), '%.0f' % p.get_height(), 
            fontsize=12, color='black', ha='center', va='bottom')
    
##########################################################################

# correlation analysis
fig, ax = plt.subplots(figsize=(20, 15))
# Create the heatmap using seaborn
sns.heatmap(df3.corr(), annot = True,cmap='coolwarm', ax=ax)
plt.show()

#Mulivariate
sns.pairplot(df3)

df4 = df3.iloc[:,[0,1,2,3,4,5,7,6]]
df4.duplicated().sum()
df4 = df4.drop_duplicates()

df4.duplicated().sum()

'''
df4.to_csv('trip_new.csv', encoding = 'utf-8')
import os
os.getcwd()
'''

num_features = df4.select_dtypes(exclude =["object"]).columns
num_features

num_features1 = ['passenger_count','PULocationID', 'DOLocationID','RatecodeID', 'payment_type']
num_features1

num_features2 = ['time_difference_minutes', 'trip_distance(m)']
num_features2

#pipeline for missing values for integer (categorical data)
from sklearn.impute import SimpleImputer

num_pipeline1 = Pipeline(steps = [('impute1', SimpleImputer(strategy='most_frequent', missing_values=0))])


#pipeline for missing values for float (continuous data)

num_pipeline2 = Pipeline(steps = [('impute2', SimpleImputer(strategy='mean'))])

preprocessor = ColumnTransformer([('most_frequent', num_pipeline1, num_features1),
                                  ('mean', num_pipeline2, num_features2)],remainder='passthrough')

imputation = preprocessor.fit(df4)

#save the impute model
joblib.dump(imputation, 'impute')

#Transform the original data
df5 = pd.DataFrame(imputation.transform(df4), columns = df4.columns)

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
                          variables=["time_difference_minutes", "trip_distance(m)","New_total_amount"])

clean = winsor.fit(df5[["time_difference_minutes", "trip_distance(m)","New_total_amount"]])

#save the winsor model
joblib.dump(clean, 'winsor')

df5[['time_difference_minutes', 'trip_distance(m)',"New_total_amount"]] = clean.fit_transform(df5[['time_difference_minutes','trip_distance(m)',"New_total_amount"]])

#Again check for outliers
df5.plot(kind = 'box', subplots = True, sharey = False, figsize = (10, 6)) 

# increase spacing between subplots
plt.subplots_adjust(wspace = 1)
plt.show()

df5.duplicated().sum()
df6= df5.drop_duplicates()
df6.duplicated().sum()
#inputs
x = df6.drop(["New_total_amount"],1)

#check for VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif1 = [variance_inflation_factor(df6.values,i) for i in list(range(df6.shape[1]))]

#output
y = df6.iloc[:,[-1]]


#Train test split
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import r2_score

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)
X_train.dtypes
X_train = X_train.astype(np.float32)
Y_train = Y_train.astype(np.float32)
X_test = X_tret.astype(np.float32)
#draft model
#p = sm.add_constant(df5.iloc[:,5,6])
base = sm.OLS(df5.iloc[:,-1],df5.iloc[:,:7]).fit()
base.summary()

######################################## Model 1
x.info()
base_1 = sm.OLS(Y_train,X_train).fit()
base_1.summary()

base1_pred_train = base_1.predict(X_train)

r2_score(Y_train,base1_pred_train)

base1_pred_test = base_1.predict(X_test)
r2_score(Y_test,base1_pred_test)

np.sqrt(np.mean((Y_train - base1_pred_train)**2))


rmse_base1_train= rmse(Y_train,base1_pred_train)
rmse_base1_train
mae_base1_train = mae(Y_train, base1_pred_train)
mae_base1_train
rmse_base1_test = rmse(Y_test,base1_pred_test)
rmse_base1_test
mae_base1_test = mae(Y_test, base1_pred_test)
mae_base1_test

# After doing winsorization for train = 0.84 and test = 0.84

######################################## Model 1-Without winsorization#####################
winsored_df = df4
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

######################################## Model 2####################################

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

rmse_lasso_train = rmse(Y_train,lasso_pred_train)
rmse_lasso_train
mae_lasso_train = mae(Y_train, lasso_pred_train)
mae_lasso_train
rmse_lasso_test = rmse(Y_test,lasso_pred_test)
rmse_lasso_test
mae_lasso_test = mae(Y_test, lasso_pred_test)
mae_lasso_test

# After doing Winsorization r2score train =0.88, test =0.88

######################################## Model 3##################################

grid2 = GridSearchCV(Ridge(), params,scoring="r2",cv = 5)
grid_model2 = grid2.fit(X_train,Y_train)
ridge_model = grid_model2.best_estimator_

ridge_pred_train = ridge_model.predict(X_train)
r2_score(Y_train,ridge_pred_train)
ridge_pred_test = ridge_model.predict(X_test)
r2_score(Y_test,ridge_pred_test)

rmse_ridge_train = rmse(Y_train,ridge_pred_train)
rmse_ridge_train
mae_ridge_train = mae(Y2_train, ridge_pred_train)
mae_ridge_train
rmse_ridge_test = rmse(Y2_test,ridge_pred_test)
rmse_ridge_test
mae_ridge_test = mae(Y2_test, ridge_pred_test)
mae_ridge_test

# After doing Winsorization our accuracy for train = 0.88 and test = 0.88
######################################## model 4###############################

grid3 = GridSearchCV(ElasticNet(), params,scoring="r2",cv = 5)
grid_model3 = grid3.fit(X_train,Y_train)
ElasticNet_model = grid_model3.best_estimator_

elastic_pred_train = ElasticNet_model.predict(X_train)
r2_score(Y_train,elastic_pred_train)
elastic_pred_test = ElasticNet_model.predict(X_test)
r2_score(Y_test,elastic_pred_test)

rmse_elas_train = rmse(Y2_train,elastic_pred_train)
rmse_elas_train
mae_elas_train = mae(Y2_train, elastic_pred_train)
mae_elas_train
rmse_elas_test = rmse(Y2_test,elastic_pred_test)
rmse_elas_test
mae_elas_test = mae(Y2_test, elastic_pred_test)
mae_elas_test

# After doing Winsorization our accuracy for train = 0.88 and test = 0.88
######################################### Model 5 --BEST MODEL #######################################

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
dt = DecisionTreeRegressor(min_samples_split=60,min_samples_leaf=3)
dt_model = dt.fit(X_train,Y_train)
dt_pred_train = dt_model.predict(X_train)
dt_pred_train = pd.DataFrame(dt_pred_train)
r2_score(Y_train,dt_pred_train)
dt_pred_test = dt_model.predict(X_test)
r2_score(Y_test,dt_pred_test)

rmse_dt_train = rmse(Y_train,dt_pred_train)
rmse_dt_train
mae_dt_train = mae(Y_train, dt_pred_train)
rmse_dt_test = rmse(Y_test,dt_pred_test)
mae_dt_test = mae(Y_test, dt_pred_test)

pickle.dump(dt_model, open('mod.pkl', 'wb'))

deploy = pickle.load(open('mod.pkl', 'rb'))

#r2 score train=0.9608962603424113 and test = 0.949337478703516

####################################model 6#######################################
from keras.models import Sequential
from keras.layers import Dense
num_data.shape[1]

x1 = x.copy()
y1 = y.copy()

x1.dtypes
x1['passenger_count'] = x1['passenger_count'].astype(np.int32)
x1['RatecodeID'] = x1['RatecodeID'].astype(np.int32)
x1['PULocationID'] = x1['PULocationID'].astype(np.int32)
x1['DOLocationID'] = x1['DOLocationID'].astype(np.int32)
x1['payment_type'] = x1['payment_type'].astype(np.int32)
x1['time_difference_minutes'] = x1['time_difference_minutes'].astype(np.float32)
x1['trip_distance(m)'] = x1['trip_distance(m)'].astype(np.float32)
y1 = y1.astype(np.float32)
y1.dtypes


X2_train, X2_test, Y2_train, Y2_test = train_test_split(x1,y1,test_size = 0.2, random_state = 0)
num_data1.dtypes
# Build neural network model
model = Sequential()
model.add(Dense(32, input_dim=x1.shape[1], activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X2_train, Y2_train, epochs=5, batch_size=32)


mod_pred_train = model.predict(X2_train)
r2_score(Y2_train,mod_pred_train)
mod_pred_test = model.predict(X2_test)
r2_score(Y2_test,mod_pred_test)

#r2 score train = 0.8932790432011705, Test= 0.8892320975898464

rmse_ann_train = rmse(Y2_train,mod_pred_train)
rmse_ann_train
mae_ann_train = mae(Y2_train, mod_pred_train)
mae_ann_train
rmse_ann_test = rmse(Y2_test,mod_pred_test)
rmse_ann_test
mae_ann_test = mae(Y2_test, mod_pred_test)
mae_ann_test

#rmse_ann_test,rmse_ann_train = 2.436021, 2.381358
#mae_ann_test,mae_ann_train =  1.210955, 1.206939

metrics = {'Models':['OLS', 'Lasso', 'Ridge', 'Elastic net', 'Decision Tree', 'ANN', 'KNN'], 
       'Test r2_score':[0.84, 0.88, 0.88, 0.88, 0.93, 0.88, 0.88],
       'Train r2_score':[0.84, 0.88, 0.88, 0.88, 0.95, 0.89, 0.92],
       'Test mae':[1.62, 1.25, 1.25, 1.25, 0.92, 1.21, 5.11],
       'Train mae':[1.62, 1.25, 1.25, 1.25, 0.80, 1.20, 3.30],
       'Test rmse':[2.65, 2.28, 2.28, 2.28, 1.67, 2.38, 2.26],
       'Train rmse':[2.65, 2.28, 2.28, 2.28, 1.46, 2.43, 1.81]}


metrics_frame = pd.DataFrame(metrics)
metrics_frame

metrics_frame.to_csv('met.csv', encoding = 'utf-8')
import os
os.getcwd()