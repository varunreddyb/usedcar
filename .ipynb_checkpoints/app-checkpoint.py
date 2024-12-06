from flask import Flask, render_template, request, jsonify, render_template_string
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import os

app = Flask(__name__)

# Define the columns based on your dataset structure
COLUMNS = ['Kilometers_Driven', 'Mileage', 'Engine', 'Power', 'Seats', 'Car_Age',
          'Location_Ahmedabad', 'Location_Bangalore', 'Location_Chennai', 'Location_Coimbatore',
          'Location_Delhi', 'Location_Hyderabad', 'Location_Jaipur', 'Location_Kochi',
          'Location_Kolkata', 'Location_Mumbai', 'Location_Pune',
          'Fuel_Type_CNG', 'Fuel_Type_Diesel', 'Fuel_Type_Electric', 'Fuel_Type_LPG', 'Fuel_Type_Petrol',
          'Transmission_Automatic', 'Transmission_Manual',
          'Owner_Type_First', 'Owner_Type_Fourth & Above', 'Owner_Type_Second', 'Owner_Type_Third',
          'Brand_Ambassador', 'Brand_Audi', 'Brand_BMW', 'Brand_Bentley', 'Brand_Chevrolet',
          'Brand_Datsun', 'Brand_Fiat', 'Brand_Force', 'Brand_Ford', 'Brand_Honda',
          'Brand_Hyundai', 'Brand_ISUZU', 'Brand_Isuzu', 'Brand_Jaguar', 'Brand_Jeep',
          'Brand_Lamborghini', 'Brand_Land', 'Brand_Mahindra', 'Brand_Maruti',
          'Brand_Mercedes-Benz', 'Brand_Mini', 'Brand_Mitsubishi', 'Brand_Nissan',
          'Brand_Porsche', 'Brand_Renault', 'Brand_Skoda', 'Brand_Smart', 'Brand_Tata',
          'Brand_Toyota', 'Brand_Volkswagen', 'Brand_Volvo']

# Load the trained model
def load_model():
    try:
        with open('usedcarpriceprediction3.pkl', 'rb') as file:
            model = pickle.load(file)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

# Initialize the model
model = load_model()



@app.route('/', methods=['GET'])
def home():
    return render_template('landing.html')

@app.route('/pre',methods=['GET'])
def renderPredictPage():
    


    return '''
    
<html>
<head>
    <title>Car Price Prediction</title>
    <style>
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        @keyframes floatAnimation {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
            100% { transform: translateY(0px); }
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
            color: white;
            padding: 40px 20px;
        }

        .container {
            width: 100%;
            max-width: 800px;
            animation: fadeInUp 0.8s ease-out;
        }

        h1 {
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 30px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
            animation: floatAnimation 3s ease-in-out infinite;
        }

        form {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            border: 1px solid rgba(255, 255, 255, 0.18);
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }

        .form-group {
            margin-bottom: 15px;
            animation: fadeInUp 0.8s ease-out;
            animation-fill-mode: both;
        }

        .form-group:nth-child(n) {
            animation-delay: calc(0.1s * var(--delay));
        }

        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 500;
            letter-spacing: 0.5px;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
        }

        input, select {
            width: 100%;
            padding: 12px;
            border: none;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(5px);
            color: #333;
            font-size: 16px;
            transition: all 0.3s ease;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }

        input:focus, select:focus {
            outline: none;
            background: white;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
            transform: translateY(-2px);
        }

        input:hover, select:hover {
            background: white;
        }

        select {
            cursor: pointer;
            appearance: none;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%23333' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'%3E%3C/polyline%3E%3C/svg%3E");
            background-repeat: no-repeat;
            background-position: right 10px center;
            background-size: 20px;
            padding-right: 40px;
        }

        .button-container {
            grid-column: 1 / -1;
            text-align: center;
            margin-top: 20px;
        }

        button {
            padding: 15px 40px;
            background: rgba(255, 255, 255, 0.2);
            color: white;
            border: none;
            border-radius: 30px;
            font-size: 18px;
            font-weight: 500;
            letter-spacing: 1px;
            cursor: pointer;
            transition: all 0.3s ease;
            backdrop-filter: blur(5px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            position: relative;
            overflow: hidden;
        }

        button:before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(
                120deg,
                transparent,
                rgba(255, 255, 255, 0.3),
                transparent
            );
            transition: 0.5s;
        }

        button:hover {
            transform: scale(1.05);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2);
        }

        button:hover:before {
            left: 100%;
        }

        @media (max-width: 768px) {
            .container {
                padding: 0 15px;
            }
            
            h1 {
                font-size: 2em;
            }

            form {
                padding: 20px;
                grid-template-columns: 1fr;
            }

            input, select {
                padding: 10px;
            }

            button {
                padding: 12px 30px;
                font-size: 16px;
            }
        }

        /* Add animation delays to form groups */
        .form-group:nth-child(1) { --delay: 1; }
        .form-group:nth-child(2) { --delay: 2; }
        .form-group:nth-child(3) { --delay: 3; }
        .form-group:nth-child(4) { --delay: 4; }
        .form-group:nth-child(5) { --delay: 5; }
        .form-group:nth-child(6) { --delay: 6; }
        .form-group:nth-child(7) { --delay: 7; }
        .form-group:nth-child(8) { --delay: 8; }
        .form-group:nth-child(9) { --delay: 9; }
        .form-group:nth-child(10) { --delay: 10; }
    </style>
    </style>
</head>
<body>
    <div class="container">
        <h1>Car Price Prediction System</h1>
        <form action="/predict" method="post">
            <div class="form-group">
                <label>Kilometers Driven:</label>
                <input type="number" name="kilometers" required placeholder="Enter kilometers driven">
            </div>
            <div class="form-group">
                <label>Mileage:</label>
                <input type="number" step="0.1" name="mileage" required placeholder="Enter mileage">
            </div>
            <div class="form-group">
                <label>Engine (cc):</label>
                <input type="number" name="engine" required placeholder="Enter engine capacity">
            </div>
            <div class="form-group">
                <label>Power (bhp):</label>
                <input type="number" step="0.1" name="power" required placeholder="Enter power">
            </div>
            <div class="form-group">
                <label>Seats:</label>
                <input type="number" name="seats" required placeholder="Enter number of seats">
            </div>
            <div class="form-group">
                <label>Car Age:</label>
                <input type="number" name="car_age" required placeholder="Enter car age">
            </div>
            <div class="form-group">
                <label>Location:</label>
                <select name="location" required>
                    <option value="" disabled selected>Select location</option>
                    <option value="Ahmedabad">Ahmedabad</option>
                    <option value="Bangalore">Bangalore</option>
                    <option value="Chennai">Chennai</option>
                    <option value="Coimbatore">Coimbatore</option>
                    <option value="Delhi">Delhi</option>
                    <option value="Hyderabad">Hyderabad</option>
                    <option value="Jaipur">Jaipur</option>
                    <option value="Kochi">Kochi</option>
                    <option value="Kolkata">Kolkata</option>
                    <option value="Mumbai">Mumbai</option>
                    <option value="Pune">Pune</option>
                </select>
            </div>
            <div class="form-group">
                <label>Fuel Type:</label>
                <select name="fuel_type" required>
                    <option value="" disabled selected>Select fuel type</option>
                    <option value="CNG">CNG</option>
                    <option value="Diesel">Diesel</option>
                    <option value="Electric">Electric</option>
                    <option value="LPG">LPG</option>
                    <option value="Petrol">Petrol</option>
                </select>
            </div>
            <div class="form-group">
                <label>Transmission:</label>
                <select name="transmission" required>
                    <option value="" disabled selected>Select transmission</option>
                    <option value="Automatic">Automatic</option>
                    <option value="Manual">Manual</option>
                </select>
            </div>
            <div class="form-group">
                <label>Owner Type:</label>
                <select name="owner_type" required>
                    <option value="" disabled selected>Select owner type</option>
                    <option value="First">First</option>
                    <option value="Second">Second</option>
                    <option value="Third">Third</option>
                    <option value="Fourth & Above">Fourth & Above</option>
                </select>
            </div>
            <div class="form-group">
                <label>Brand:</label>
                <select name="brand" required>
                    <option value="" disabled selected>Select brand</option>
                    <option value="Ambassador">Ambassador</option>
                    <option value="Audi">Audi</option>
                    <option value="BMW">BMW</option>
                    <option value="Honda">Honda</option>
                    <option value="Hyundai">Hyundai</option>
                    <option value="Mahindra">Mahindra</option>
                    <option value="Maruti">Maruti</option>
                    <option value="Mercedes-Benz">Mercedes-Benz</option>
                    <option value="Tata">Tata</option>
                    <option value="Toyota">Toyota</option>
                </select>
            </div>
            <div class="button-container">
                <button type="submit">Predict Price</button>
            </div>
        </form>
    </div>
</body>
</html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        # Create a DataFrame with all features initialized to 0
        input_df = pd.DataFrame(0, index=[0], columns=COLUMNS)
        
        # Fill numerical values
        input_df['Kilometers_Driven'] = float(request.form['kilometers'])
        input_df['Mileage'] = float(request.form['mileage'])
        input_df['Engine'] = float(request.form['engine'])
        input_df['Power'] = float(request.form['power'])
        input_df['Seats'] = int(request.form['seats'])
        input_df['Car_Age'] = int(request.form['car_age'])
        
        # Set categorical variables
        location_col = f"Location_{request.form['location']}"
        fuel_type_col = f"Fuel_Type_{request.form['fuel_type']}"
        transmission_col = f"Transmission_{request.form['transmission']}"
        owner_type_col = f"Owner_Type_{request.form['owner_type']}"
        brand_col = f"Brand_{request.form['brand']}"
        
        # Set the one-hot encoded columns
        for col in [location_col, fuel_type_col, transmission_col, owner_type_col, brand_col]:
            if col in input_df.columns:
                input_df[col] = 1
        
        # Make prediction
        prediction = model.predict(input_df)
        
        return render_template_string('''
            <!DOCTYPE html>
<html>
<head>
    <title>Prediction Result</title>
    <style>
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        @keyframes floatAnimation {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
            100% { transform: translateY(0px); }
        }

        @keyframes shimmer {
            0% { background-position: -1000px 0; }
            100% { background-position: 1000px 0; }
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
            color: white;
        }

        .container {
            text-align: center;
            animation: fadeInUp 0.8s ease-out;
            padding: 20px;
        }

        h1 {
            font-size: 2.5em;
            margin-bottom: 30px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
            animation: floatAnimation 3s ease-in-out infinite;
        }

        .result {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            padding: 30px 50px;
            border-radius: 15px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            border: 1px solid rgba(255, 255, 255, 0.18);
            margin: 20px 0;
            transform: translateY(0);
            transition: transform 0.3s ease;
        }

        .result:hover {
            transform: translateY(-5px);
        }

        .price {
            font-size: 2.8em;
            font-weight: bold;
            background: linear-gradient(90deg, #ffd700, #fff, #ffd700);
            background-size: 200% 100%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: shimmer 3s infinite linear;
            margin: 10px 0;
        }

        .back-button {
            margin-top: 30px;
            perspective: 1000px;
        }

        .back-button a {
            display: inline-block;
            padding: 15px 30px;
            background: rgba(255, 255, 255, 0.2);
            color: white;
            text-decoration: none;
            border-radius: 30px;
            font-weight: 500;
            letter-spacing: 1px;
            backdrop-filter: blur(5px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .back-button a:before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(
                120deg,
                transparent,
                rgba(255, 255, 255, 0.3),
                transparent
            );
            transition: 0.5s;
        }

        .back-button a:hover {
            transform: scale(1.05);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        }

        .back-button a:hover:before {
            left: 100%;
        }

        .currency-symbol {
            font-size: 0.7em;
            vertical-align: super;
        }

        .unit {
            font-size: 0.5em;
            opacity: 0.8;
            margin-left: 5px;
        }

        @media (max-width: 600px) {
            .container {
                padding: 10px;
            }
            
            h1 {
                font-size: 2em;
            }

            .price {
                font-size: 2.2em;
            }

            .result {
                padding: 20px 30px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Prediction Result</h1>
        <div class="result">
            <div class="price">
                <span class="currency-symbol">₹</span>
                {{ "{:,.2f}".format(prediction[0]) }}Lakhs
                
            </div>
        </div>
        <div class="back-button">
            <a href="/">Back to Prediction Form</a>
        </div>
    </div>
</body>
</html>
        ''', prediction=prediction)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        # Get JSON data
        data = request.get_json()
        
        # Create input DataFrame with all features initialized to 0
        input_df = pd.DataFrame(0, index=[0], columns=COLUMNS)
        
        # Fill numerical values
        numerical_columns = {
            'Kilometers_Driven': float,
            'Mileage': float,
            'Engine': float,
            'Power': float,
            'Seats': int,
            'Car_Age': int
        }
        
        for col, dtype in numerical_columns.items():
            if col in data:
                input_df[col] = dtype(data[col])
        
        # Handle categorical variables
        if 'Location' in data:
            col = f"Location_{data['Location']}"
            if col in input_df.columns:
                input_df[col] = 1
                
        if 'Fuel_Type' in data:
            col = f"Fuel_Type_{data['Fuel_Type']}"
            if col in input_df.columns:
                input_df[col] = 1
                
        if 'Transmission' in data:
            col = f"Transmission_{data['Transmission']}"
            if col in input_df.columns:
                input_df[col] = 1
                
        if 'Owner_Type' in data:
            col = f"Owner_Type_{data['Owner_Type']}"
            if col in input_df.columns:
                input_df[col] = 1
                
        if 'Brand' in data:
            col = f"Brand_{data['Brand']}"
            if col in input_df.columns:
                input_df[col] = 1
        
        # Make prediction
        prediction = model.predict(input_df)
        
        return jsonify({
            'predicted_price': float(prediction[0]),
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

if __name__ == '__main__':
    app.run(debug=True)