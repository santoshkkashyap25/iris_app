import os
import logging
from flask import Flask, request, render_template, redirect, url_for, flash
from config import config_by_name
from model import load_iris_model, predict_iris, get_feature_names

# Determine the configuration based on environment variable (e.g., FLASK_ENV=production)
env_config = os.getenv('FLASK_ENV', 'default')
Config = config_by_name[env_config]

app = Flask(__name__)
app.config.from_object(Config)
app.secret_key = 'your_super_secret_key_here' # Needed for flash messages

# Configure logging for the Flask app
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(app.config['LOG_FILE']),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

# Load the model when the application starts
with app.app_context(): # Ensure app context for config access
    try:
        load_iris_model(app.config['MODEL_PATH'])
    except (FileNotFoundError, RuntimeError) as e:
        logger.critical(f"Failed to load model at application startup: {e}")
    
    # Get feature names and ranges for home page
    try:
        app.feature_names = get_feature_names()
        app.feature_ranges = app.config['IRIS_FEATURE_RANGES']
    except Exception as e:
        logger.error(f"Could not load feature names or ranges: {e}")
        app.feature_names = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
        app.feature_ranges = {} # Fallback

@app.route('/')
def home():
    """Renders the home page with the input form."""
    logger.info("Serving home page.")
    return render_template('home.html', 
                           feature_names=app.feature_names, 
                           feature_ranges=app.feature_ranges)

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request from the form."""
    logger.info("Received prediction request.")
    
    features = []
    # Use feature_names from model to get data in correct order and link to ranges
    input_keys = ['a', 'b', 'c', 'd'] # Corresponding to form inputs
    
    try:
        for i, key in enumerate(input_keys):
            value = float(request.form[key])
            feature_name = app.feature_names[i]
            
            # Basic range validation (can be more sophisticated)
            if feature_name in app.feature_ranges:
                min_val = app.feature_ranges[feature_name]['min']
                max_val = app.feature_ranges[feature_name]['max']
                if not (min_val <= value <= max_val):
                    flash(f"Warning: {feature_name.title()} ({value} cm) is outside typical range ({min_val}-{max_val} cm).", 'warning')
                    logger.warning(f"Input value {value} for {feature_name} is out of typical range.")

            features.append(value)

        logger.info(f"Input features for prediction: {features}")

        predicted_class_label, probabilities = predict_iris(features)

        if predicted_class_label == -1:
            flash("Error: Model not loaded or prediction failed. Please check server logs.", 'error')
            logger.error("Prediction failed due to model issues.")
            return redirect(url_for('home'))

        # Get species info from config
        species_info = app.config['IRIS_SPECIES_MAP'].get(predicted_class_label, {
            "name": "Unknown Species",
            "image": "default.png", # Provide a default image
            "description": "Could not identify species due to an unexpected prediction result."
        })

        # Determine confidence message
        main_probability = probabilities[predicted_class_label]
        confidence_message = ""
        if main_probability >= 0.95:
            confidence_message = "This is a very high confidence prediction!"
        elif main_probability >= 0.8:
            confidence_message = "This is a strong prediction."
        elif main_probability >= 0.6:
            confidence_message = "This is a moderate prediction, some ambiguity might exist."
        else:
            confidence_message = "This prediction has lower confidence, indicating feature values might be near decision boundaries."


        # Prepare probabilities for display
        probabilities_display = []
        for class_label, prob in enumerate(probabilities):
            species_name = app.config['IRIS_SPECIES_MAP'].get(class_label, {}).get("name", f"Class {class_label}")
            probabilities_display.append({
                "name": species_name,
                "probability": f"{prob:.2%}" # Format as percentage
            })

        logger.info(f"Predicted species: {species_info['name']}, Confidence: {main_probability:.2%}")
        return render_template('after.html',
                               species_info=species_info,
                               probabilities=probabilities_display,
                               confidence_message=confidence_message,
                               predicted_features=features, # Pass back features for context
                               feature_names=app.feature_names # Pass feature names for context
                               )

    except ValueError:
        logger.error("Invalid input: Non-numeric values provided for features.")
        flash("Please enter valid numeric values for all flower measurements.", 'error')
        return redirect(url_for('home'))
    except Exception as e:
        logger.critical(f"An unhandled error occurred during prediction: {e}")
        flash(f"An unexpected error occurred: {e}. Please try again.", 'error')
        return redirect(url_for('home'))

# Custom error handler for 404 Not Found
@app.errorhandler(404)
def page_not_found(e):
    logger.warning(f"404 Not Found: {request.url}")
    return render_template('404.html'), 404


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)