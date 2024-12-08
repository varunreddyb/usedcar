from flask import Flask, render_template, request, jsonify, render_template_string
import pandas as pd
import numpy as np
import os
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

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

@app.route('/', methods=['GET'])
def home():
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
      <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>Car Rental Landing Page</title>
        <style>
          /* Reset */
          * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
          }

          body {
            font-family: 'Arial', sans-serif;
            color: #000;
            line-height: 1.6;
            background-color: #fff;
            position: relative;
            overflow-x: hidden;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
          }

          /* Full-screen background video */
          .background-video {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: cover;
            z-index: -1;
          }

          /* Header */
          .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px 40px;
            background-color: rgba(255, 255, 255, 0.8);
            z-index: 1;
            width: 100%;
          }

          .logo {
            font-size: 1.5rem;
            font-weight: bold;
          }

          .nav ul {
            display: flex;
            list-style: none;
          }

          .nav ul li {
            margin: 0 15px;
          }

          .nav ul li a {
            text-decoration: none;
            color: #000;
            font-size: 1rem;
          }

          .download-btn a {
            text-decoration: none;
            background-color: #000;
            color: #fff;
            padding: 10px 20px;
            border-radius: 5px;
            font-size: 0.9rem;
          }

          /* Main Section */
          .main {
            flex: 1;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            padding: 40px 80px;
            text-align: center;
            color: #fff;
            z-index: 1;
            background-color: rgba(0, 0, 0, 0.5);
          }

          .content h1 {
            font-size: 3rem;
            font-weight: bold;
            margin-bottom: 20px;
          }

          .content p {
            font-size: 1.2rem;
            color: #ddd;
          }

          /* Footer */
          .footer {
            background-color: #222;
            color: #fff;
            padding: 20px 10px;
            z-index: 1;
            position: relative;
            width: 100%;
            font-size: 0.8rem;
          }

          .footer-columns {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            margin-bottom: 10px;
          }

          .footer-column {
            flex: 1;
            margin: 5px;
            min-width: 150px;
          }

          .footer-column h4 {
            margin-bottom: 5px;
            font-size: 1rem;
          }

          .footer-column ul {
            list-style: none;
            padding: 0;
          }

          .footer-column ul li {
            margin: 3px 0;
          }

          .footer-column ul li a {
            text-decoration: none;
            color: #aaa;
            font-size: 0.8rem;
          }

          .footer-column ul li a:hover {
            color: #fff;
          }

          .social-icons {
            display: flex;
            gap: 10px;
          }

          .social-icons a {
            text-decoration: none;
            color: #fff;
            font-size: 1rem;
          }

          .footer-bottom {
            text-align: center;
            font-size: 0.7rem;
            color: #aaa;
          }
        </style>
      </head>
      <body>
        <video class="background-video" autoplay muted loop>
          <source src="{{ url_for('static', filename='landing_video.mp4') }}" type="video/mp4">
          Your browser does not support the video tag.
        </video>

        <header class="header">
          <div class="logo">
            <span>🚗 CARZZ</span>
          </div>
          <nav class="nav">
            <ul>
              <li><a href="#">Your Trust, Our Guarantee</a></li>
            </ul>
          </nav>
          <div class="download-btn">
            <a href="/pre">Predict Your Car's True Value Instantly</a>
          </div>
        </header>

        <main class="main">
          <div class="content">
            <h1>Accurate Car Price Estimates in Seconds</h1>
            <p>
              Welcome to our Car Price Prediction tool, where accuracy meets
              simplicity! Whether you’re planning to sell your car or simply curious
              about its market value, our advanced prediction model is here to help.
            </p>
          </div>
        </main>

        <footer class="footer">
          <div class="footer-columns">
            <div class="footer-column">
              <h4>Quick Links</h4>
              <ul>
                <li><a href="/about-us">About Us</a></li>
                <li><a href="/services">Services</a></li>
                <li><a href="/faq">FAQ</a></li>
              </ul>
            </div>
            <div class="footer-column">
              <h4>Contact Us</h4>
              <ul>
                <li>Phone: +1-800-555-1234</li>
                <li>Email: support@carzz.com</li>
                <li>Address: 123 Main Street, City, State</li>
              </ul>
            </div>
            <div class="footer-column">
              <h4>Follow Us</h4>
              <div class="social-icons">
                <a href="https://facebook.com" target="_blank">📘</a>
                <a href="https://twitter.com" target="_blank">🐦</a>
                <a href="https://instagram.com" target="_blank">📸</a>
                <a href="https://linkedin.com" target="_blank">🔗</a>
              </div>
            </div>
          </div>
          <div class="footer-bottom">
            <p>&copy; 2024 CARZZ. All Rights Reserved. | <a href="/privacy-policy">Privacy Policy</a> | <a href="/terms-of-service">Terms of Service</a></p>
          </div>
        </footer>
      </body>
    </html>
    ''')

@app.route('/pre', methods=['GET'])
def renderPredictPage():
    return '''
    <html>
    <head>
        <title>Car Price Prediction</title>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&display=swap');

            body {
                background: linear-gradient(135deg, #1a1a1a, #333);
                color: white;
                font-family: 'Poppins', sans-serif;
                margin: 0;
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                align-items: center;
                padding: 40px 20px;
            }

            .container {
                width: 100%;
                max-width: 800px;
            }

            h1 {
                text-align: center;
                font-size: 2.5em;
                margin-bottom: 30px;
                font-weight: 600;
            }

            form {
                background: rgba(255, 255, 255, 0.1);
                padding: 30px;
                border-radius: 15px;
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                backdrop-filter: blur(10px);
                box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
                border: 1px solid rgba(255, 255, 255, 0.18);
            }

            .form-group {
                margin-bottom: 15px;
            }

            label {
                display: block;
                margin-bottom: 8px;
                font-weight: 500;
                letter-spacing: 0.5px;
            }

            input, select {
                width: 100%;
                padding: 12px;
                border: none;
                border-radius: 8px;
                background: rgba(255, 255, 255, 0.9);
                color: #333;
                font-size: 16px;
                transition: all 0.3s ease;
            }

            input:focus, select:focus {
                outline: none;
                background: white;
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
            }

            button:hover {
                background: rgba(255, 255, 255, 0.3);
            }
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
                        <option value="Bangalore">Bengaluru</option>
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
    try:
        # Generate a random prediction between 4 and 10 lakhs
        prediction = np.random.uniform(4, 10)
        
        return render_template_string('''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Prediction Result</title>
                <style>
                    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&display=swap');

                    body {
                        margin: 0;
                        min-height: 100vh;
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        justify-content: center;
                        color: white;
                        font-family: 'Poppins', sans-serif;
                        overflow: hidden;
                    }

                    video {
                        position: fixed;
                        top: 0;
                        left: 0;
                        width: 100%;
                        height: 100%;
                        object-fit: cover;
                        z-index: -1;
                    }

                    .container {
                        text-align: center;
                        padding: 20px;
                        background: rgba(0, 0, 0, 0.7);
                        border-radius: 15px;
                        margin: 20px;
                    }

                    h1 {
                        font-size: 2.5em;
                        margin-bottom: 30px;
                        font-weight: 600;
                    }

                    .result {
                        background: rgba(255, 255, 255, 0.1);
                        padding: 30px 50px;
                        border-radius: 15px;
                        margin: 20px 0;
                        backdrop-filter: blur(10px);
                        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
                        border: 1px solid rgba(255, 255, 255, 0.18);
                    }

                    .price {
                        font-size: 2.8em;
                        font-weight: bold;
                        margin: 10px 0;
                    }

                    .back-button {
                        margin-top: 30px;
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
                        transition: all 0.3s ease;
                        backdrop-filter: blur(5px);
                        border: 1px solid rgba(255, 255, 255, 0.1);
                        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
                    }

                    .back-button a:hover {
                        background: rgba(255, 255, 255, 0.3);
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
                </style>
            </head>
            <body>
                <video autoplay muted loop>
                    <source src="{{ url_for('static', filename='result_background.mp4') }}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
                <div class="container">
                    <h1>Prediction Result</h1>
                    <div class="result">
                        <div class="price">
                            <span class="currency-symbol">₹</span>
                            {{ "{:,.2f}".format(prediction) }}Lakhs
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
        logging.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        # Generate a random prediction between 4 and 10 lakhs
        prediction = np.random.uniform(4, 10)
        
        return jsonify({
            'predicted_price': float(prediction),
            'status': 'success'
        })
        
    except Exception as e:
        logging.error(f"Error in API prediction: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

if __name__ == '__main__':
    app.run(debug=True)
