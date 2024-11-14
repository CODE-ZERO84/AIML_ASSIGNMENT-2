from flask import Flask, request, render_template_string
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and scaler (ensure the paths are correct)
model = joblib.load('best_model.pkl')  # Update the path if necessary
scaler = joblib.load('scaler.pkl')  # Load the scaler (if it's saved)

# Define the HTML form inside the Python code for simplicity
form_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prediction Form</title>
    <style>
        body {
            background-color: rgb(255,0,0);  /* Light green background */
            
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }
        .container {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.1);
            width: 400px;
        }
        h2 {
            text-align: center;
            color: #333;
        }
        label {
            font-size: 16px;
            margin-bottom: 5px;
            display: block;
            color: #333;
        }
        input[type="number"] {
            width: 100%;
            padding: 10px;
            margin: 10px 0;
            border: 1px solid #ccc;
            border-radius: 5px;
            font-size: 16px;
        }
        input[type="submit"] {
            background-color: #007bff;  /* Blue button */
            color: white;
            border: none;
            padding: 12px 20px;
            text-align: center;
            font-size: 16px;
            border-radius: 5px;
            cursor: pointer;
            width: 100%;
        }
        input[type="submit"]:hover {
            background-color: #0056b3;  /* Darker blue on hover */
        }
        .prediction-text {
            text-align: center;
            font-size: 18px;
            margin-top: 20px;
            font-weight: bold;
        }
    </style>
</head>
<body>

    <div class="container">
        <h2>Enter Feature Values for Prediction</h2>
        <form method="POST" action="/predict">
            <label for="feature1">GDP per capita (in USD):</label>
            <input type="number" step="any" name="feature1" required><br>

            <label for="feature2">Literacy Rate (in %):</label>
            <input type="number" step="any" name="feature2" required><br>

            <input type="submit" value="Predict">
        </form>

        {% if prediction_text %}
            <div class="prediction-text">
                <p>{{ prediction_text }}</p>
            </div>
        {% endif %}
    </div>

</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(form_html)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input features from the form
        feature1 = float(request.form['feature1'])
        feature2 = float(request.form['feature2'])

        # Print the received input values for debugging
        print(f"Received input: {feature1}, {feature2}")

        # Combine the features into an array for the model
        features = np.array([feature1, feature2]).reshape(1, -1)

        # Scale the features using the loaded scaler
        features_scaled = scaler.transform(features)

        # Print the scaled features to check the scaling process
        print(f"Scaled features: {features_scaled}")

        # Make the prediction
        prediction = model.predict(features_scaled)[0]

        # Translate prediction to readable result
        result = "High" if prediction == 1 else "Low"

        return render_template_string(form_html, prediction_text=f"Predicted Food Waste Category: {result}")

    except Exception as e:
        # Print the detailed error message for debugging
        print(f"Error: {str(e)}")
        return render_template_string(form_html, prediction_text="Error occurred. Please check your inputs.")

if __name__ == '__main__':
    app.run(debug=True)
