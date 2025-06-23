# Iris Flower Species Predictor

## Project Overview

This is a web application built with **Flask** that predicts the species of an Iris flower based on its sepal and petal measurements. It utilizes a pre-trained **Logistic Regression** machine learning model to provide accurate predictions along with confidence scores. The application aims to offer an interactive and informative experience for understanding the classic Iris dataset.

## Live Demo

Experience the Iris Flower Species Predictor live here:  
[https://flotect.onrender.com](https://flotect.onrender.com)

## Features

- **Interactive Input**: User-friendly sliders for entering sepal and petal measurements (length and width in centimeters).
- **Real-time Value Display**: See the selected measurement values update as you drag the sliders.
- **Accurate Prediction**: Predicts one of three Iris species: *Iris-setosa*, *Iris-versicolor*, or *Iris-virginica*.
- **Prediction Confidence**: Displays probability scores for each possible species, indicating the model's confidence.
- **Contextual Information**: Provides a brief description of the predicted Iris species.
- **Input Guidance**: Sliders are pre-configured with realistic measurement ranges based on the Iris dataset.
- **Robust Error Handling**: Catches non-numeric input errors and shows user-friendly messages.
- **Logging**: Basic logging implemented for monitoring and debugging.

## Technologies Used

- **Flask**: Web framework for the application.
- **Scikit-learn**: For the machine learning model (Logistic Regression).
- **NumPy & Pandas**: For data handling and numerical operations.
- **Gunicorn**: WSGI HTTP server for production.
- **Render**: Platform as a Service (PaaS) used for deployment.

## Local Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/santoshkkashyap25/iris_app.git
cd iris_app
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

- On macOS/Linux:

```bash
source venv/bin/activate
```

- On Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Train and Save the Model

```bash
python train_model.py
```

This will generate `iris.pkl` in your project root directory.

### 6. Run the Flask Application

```bash
python app.py
```

### 7. Access the Application

Open your browser and go to:  
[http://127.0.0.1:5000/](http://127.0.0.1:5000/) or [http://localhost:5000/](http://localhost:5000/)

## Usage

- **Home Page (`/`)**: Enter the four required measurements (Sepal Length, Sepal Width, Petal Length, Petal Width) using sliders.
- **Prediction (`/predict`)**: Click the **Predict Species** button.
- **Result Page (`/after`)**:
  - Displays the predicted Iris species.
  - Shows the model's confidence level.
  - Lists probabilities for all three species.
  - Recaps the entered measurements.
  - Provides information about the predicted species.
- Click **Try Another Prediction** to return to the home page.

## Project Structure

```
iris_app/
├── app.py              # Main Flask application
├── config.py           # Configuration settings and mappings
├── model.py            # Model loading and prediction logic
├── templates/          # HTML templates
│   ├── home.html       # Input form
│   ├── after.html      # Result page
│   └── 404.html        # Custom error page
├── train_model.py      # Model training script
├── iris.pkl            # Trained model file
├── requirements.txt    # Python dependencies
├── .gitignore          # Git ignore file
├── Procfile            # Render deployment instructions
└── app.log             # Log file (ignored by Git)
```

## Deployment

The application is deployed on **Render** using **Gunicorn** as the WSGI server. The `Procfile` tells Render how to run the app. The trained model (`iris.pkl`) is included in the repository and loaded during deployment.

## Future Improvements

- Implement more advanced input validation (e.g., ensure petal length ≤ sepal length).
- Add a **Reset** button to the home page.
- Integrate a small plot showing input measurements relative to typical species ranges.
- Explore different ML models (SVM, Decision Trees) and allow user selection.