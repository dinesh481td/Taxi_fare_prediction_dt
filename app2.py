from flask import Flask, render_template, request
import re
import pandas as pd
import copy
import pickle
import joblib

deploy = pickle.load(open('C:/Users/Dinesh T/Downloads/Taxi_flask/mod.pkl','rb'))
impute = joblib.load('C:/Users/Dinesh T/Downloads/Taxi_flask/impute')
winsor = joblib.load('C:/Users/Dinesh T/Downloads/Taxi_flask/winsor')
'''
# connecting to sql by creating sqlachemy engine
from sqlalchemy import create_engine
engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                       .format(user = "root",#user
                               pw = "user1", # passwrd
                               db = "Taxi"))
'''
def decision_tree(data_new):
    clean1 = pd.DataFrame(impute.transform(data_new), columns = data_new.select_dtypes(exclude = ['object']).columns)
    clean1[['time_difference_minutes', 'trip_distance(m)']] = winsor.transform(clean1[['time_difference_minutes', 'trip_distance(m)']])
    prediction = pd.DataFrame(deploy.predict(clean1), columns = ['Output'])
    final_data = pd.concat([data_new,prediction], axis = 1)
    return(final_data)
    
            
#define flask
app = Flask(__name__)



@app.route('/')
def home():
    
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    passenger_count = request.form['Passenger_Count']
    ratecodeid = request.form['RatecodeID']
    pulocationid = request.form['PULocationID']
    dolocationid = request.form['DOLocationID']
    payment_type = request.form['Payment_type']
    time_difference_minutes = request.form['Time_difference_minutes']
    trip_distance = request.form['Trip_distance']
    
    # convert the input to the format expected by the model
    input_data = [[int(passenger_count), int(ratecodeid), int(pulocationid), int(dolocationid), int(payment_type), float(time_difference_minutes), float(trip_distance)]]
    
    # do the prediction using the loaded model
    prediction = deploy.predict(input_data)
    
    return render_template('index1.html', prediction=prediction)



@app.route('/success', methods = ['POST'])
def success():
    if request.method == 'POST' :
        f = request.files['file']
        data_new = pd.read_excel(f)
       
        final_data = decision_tree(data_new)

        f#  inal_data.to_sql('taxi_pred', con = engine, if_exists = 'replace', chunksize = 6500000, index= False)
        
        
       
        return render_template("new.html", Y = final_data.to_html(justify = 'center'))


if __name__=='__main__':
    app.run(debug = True)
