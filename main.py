from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

# Load the saved model from the file
loaded_model = joblib.load('lr_model.pkl')
loaded_model2 = joblib.load('RF_model.pkl')
loaded_model3 = joblib.load('best_model.pkl')

# Create a Flask web application
app = Flask(__name__)


# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')


# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data as a JSON object
        input_data = request.get_json()

        # Ensure that the input_data contains the expected features
        expected_features = ['Age', 'Gender', 'Circadian_Rhythm', 'Exercise_Hours', 'Diet_Caffeine', 'Stress_Level',
                             'Sleep_Duration']
        if not all(feature in input_data for feature in expected_features):
            return jsonify({'error': 'Invalid input format. Expected features: ' + ', '.join(expected_features)}), 400

        # Prepare the input data for prediction
        input_list = [
            input_data['Age'],
            input_data['Gender'],
            input_data['Circadian_Rhythm'],
            input_data['Exercise_Hours'],
            input_data['Diet_Caffeine'],
            input_data['Stress_Level'],
            input_data['Sleep_Duration']
        ]

        input_df = pd.DataFrame([input_list])

        # Make a prediction using the loaded model
        predictions = loaded_model.predict(input_df)

        # Map numeric predictions to labels
        label_mapping = {0: 'Excellent', 1: 'Good', 2: 'Poor'}
        predicted_labels = [label_mapping[prediction] for prediction in predictions]

        # Return the predicted labels as JSON response
        return jsonify({'predictions': predicted_labels})

    except Exception as e:
        return jsonify({'error': 'An error occurred: ' + str(e)}), 500


@app.route('/predict_heart_rate', methods=['POST'])
def predict_heart_rate():
    try:
        # Get the input data as a JSON object
        input_data = request.get_json()

        # Ensure that the input_data contains the expected features
        expected_features = ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5']
        if not all(feature in input_data for feature in expected_features):
            return jsonify({'error': 'Invalid input format. Expected features: ' + ', '.join(expected_features)}), 400

        # Prepare the input data for prediction
        input_list = [
            input_data['Feature1'],
            input_data['Feature2'],
            input_data['Feature3'],
            input_data['Feature4'],
            input_data['Feature5']
        ]

        input_df = pd.DataFrame([input_list])

        # Make a prediction using the loaded model
        predictions = loaded_model2.predict(input_df)

        # Map numeric predictions to labels
        label_mapping = {0: 'Abnormal', 1: 'Normal'}
        predicted_labels = [label_mapping[prediction] for prediction in predictions]

        # Return the predicted labels as JSON response
        return jsonify({'predictions': predicted_labels})

    except Exception as e:
        return jsonify({'error': 'An error occurred: ' + str(e)}), 500


# Define a route for making predictions using the loaded model
@app.route('/predict_chronic_risk', methods=['POST'])
def predict_chronic_risk():
    try:
        # Get the input data as a JSON object
        input_data = request.get_json()

        # Ensure that the input_data contains the expected features
        expected_features = ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5', 'Feature6', 'Feature7',
                             'Feature8', 'Feature9']
        if not all(feature in input_data for feature in expected_features):
            return jsonify({'error': 'Invalid input format. Expected features: ' + ', '.join(expected_features)}), 400

        # Prepare the input data for prediction
        input_list = [
            input_data['Feature1'],
            input_data['Feature2'],
            input_data['Feature3'],
            input_data['Feature4'],
            input_data['Feature5'],
            input_data['Feature6'],
            input_data['Feature7'],
            input_data['Feature8'],
            input_data['Feature9']
        ]

        input_df = pd.DataFrame([input_list])

        # Make a prediction using the loaded model
        predictions = loaded_model3.predict(input_df)

        # Map numeric predictions to labels
        label_mapping = {0: 'There is a Risk of Chronic Kidney Disease', 1: 'There is no risk of Chronic Kidney Disease'}
        predicted_labels = [label_mapping[prediction] for prediction in predictions]

        # Return the predicted labels as JSON response
        return jsonify({'predictions': predicted_labels})

    except Exception as e:
        return jsonify({'error': 'An error occurred: ' + str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
