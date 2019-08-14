from flask import Flask, jsonify, request, render_template
import joblib
import pandas as pd

app = Flask(__name__, static_url_path='')

df = pd.read_csv('data_clean.csv')

y = df['price']
del df['price']
df = df.drop('Unnamed: 0', 1)
x = df.values
y = y.values

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/eda')
def eda():
    return render_template('eda.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/result', methods = ['POST'])
def result():
    bedrooms = int(request.form['bedrooms'])
    bathrooms = int(request.form['bathrooms'])
    sqft_living = int(request.form['sqft_living'])
    sqft_lot = int(request.form['sqft_lot'])
    floors = int(request.form['floors'])
    waterfront = int(request.form['waterfront'])
    condition = int(request.form['condition'])
    grade = int(request.form['grade'])
    sqft_above = int(request.form['sqft_above'])
    sqft_basement = int(request.form['sqft_basement'])
    yr_built = int(request.form['yr_built'])
    yr_renovated = int(request.form['yr_renovated'])
    lat = int(request.form['lat'])
    long = int(request.form['long'])
    sqft_living15 = int(request.form['sqft_living15'])
    sqft_lot15 = int(request.form['sqft_lot15'])

    array = pd.Series([bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, condition, grade, sqft_above, 
        sqft_basement, yr_built, yr_renovated, lat, long, sqft_living15, sqft_lot15])
    values = array.values

    def place_value(number): 
        return ("{:,}".format(number)) 

    price_LR = place_value(int(model_LR.predict([values])))
    score_LR = round(model_LR.score(x, y) * 100, 2)
    price_RF = place_value(int(model_RF.predict([values])))
    score_RF = round(model_RF.score(x, y) * 100, 2)
    price_GB = place_value(int(model_GB.predict([values])))
    score_GB = round(model_GB.score(x, y) * 100, 2) 
    
    return render_template('result.html', 
        price_LR = price_LR, score_LR = score_LR, 
        price_RF = price_RF, score_RF = score_RF,
        price_GB = price_GB, score_GB =  score_GB
        )


if __name__ == '__main__':
    model_LR = joblib.load('model_LR.joblib')  
    model_RF = joblib.load('model_RF.joblib')
    model_GB = joblib.load('model_GB.joblib')
    app.run(debug = True, port = 5001)   