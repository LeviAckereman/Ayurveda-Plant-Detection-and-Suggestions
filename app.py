from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3

# Import the login blueprint
from login import login_blueprint  # Assuming login.py is in the same directory

# Initialize Flask app
app = Flask(__name__)

# Register the blueprint for login-related routes
app.register_blueprint(login_blueprint, url_prefix='/login')
app.config['UPLOAD_FOLDER'] = 'static/images/uploads/'
app.secret_key = os.urandom(24)  # Required for session to work

# Load the VGG-19 model
model = load_model('D:\\ADITYA\\BE PROJECT\\Ayurveda Website\\plant_model.h5')

# Load the Ayurveda data from Excel
ayurveda_data = pd.read_excel("D:\\ADITYA\\BE PROJECT\\Ayurveda Website\\Medicinal Plants Dataset.xlsx")
ayurveda_data.columns = ayurveda_data.columns.str.strip()  # Clean any extra spaces in the column names

# Preprocess the image for VGG-19
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Predict the plant
def predict_plant(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    
    # Get the predicted class and the associated confidence score
    predicted_class = np.argmax(predictions, axis=1)
    predicted_confidence = np.max(predictions)  # Confidence score for the predicted class
    
    # Define a threshold for the model's confidence (e.g., 50% confidence)
    confidence_threshold = 0.5
    
    if predicted_confidence < confidence_threshold:
        # If the confidence is lower than the threshold, assume it's not a plant
        return None, predicted_confidence  # Return None to indicate no plant detected
    
    return predicted_class[0], predicted_confidence

# Get plant information based on prediction index
def get_plant_info(prediction_index):
    try:
        # Get the row corresponding to the predicted class index
        plant_row = ayurveda_data.iloc[prediction_index]
        
        # Safely get plant information, falling back to "Not Available" if a column is missing
        medicinal_part = plant_row.get("Medicinal Part", "Not Available")
        medicinal_uses = plant_row.get("Medicinal Uses", "Not Available")
        
        return {
            "Name": plant_row["Name"],
            "Botanical Name": plant_row["Botanical Name"],
            "Medicinal Part": medicinal_part,
            "Medicinal Uses": medicinal_uses
        }
    except Exception as e:
        print(f"Error retrieving plant info: {e}")
        return {"Name": "Not Available", "Botanical Name": "Not Available", "Medicinal Part": "Not Available", "Medicinal Uses": "Not Available"}

# Function to validate if the uploaded file is a plant image
def is_plant_image(file_path):
    try:
        # Open the image using PIL to ensure it's a valid image
        img = Image.open(file_path)
        img.verify()  # This will check if the image is valid
        return True
    except (IOError, ValueError):
        # If it's not a valid image or is corrupt, return False
        return False

# Database setup function (create the users table if it doesn't exist)
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE,
            password TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Initialize the database
init_db()

# Home route (this will be the page users see after login)
@app.route('/home')
def home():
    if 'logged_in' in session and session['logged_in']:
        username = session['username']  # Get the username from the session
        return render_template('index.html', username=username)  # Render the home page
    else:
        return redirect(url_for('index'))  # If not logged in, redirect to login/signup page

# Login and Signup route
from werkzeug.security import generate_password_hash, check_password_hash

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle Login
        if 'login' in request.form:
            username = request.form['username']
            password = request.form['password']
            action = request.form.get('action')  # To differentiate between login and signup actions

            # Fetch user from the database
            conn = sqlite3.connect('users.db')
            c = conn.cursor()

        if action == 'Login':
            # Login Logic
            c.execute('SELECT * FROM users WHERE username = ?', (username,))
            user = c.fetchone()
            conn.close()

            # Print the user record for debugging
            print(f"User record fetched: {user}")

            # Check if user exists and the password is correct
            if user and check_password_hash(user[2], password):  # user[2] is the hashed password
                session['logged_in'] = True
                session['username'] = username  # Store username in session
                return redirect(url_for('home'))  # Redirect to the home page after login
            else:
                error_message = "Invalid username or password!"
                return render_template('index.html', error_message=error_message)
        
        # Handle Signup
        elif 'signup' in request.form:
            username = request.form['username']
            password = request.form['password']
            
            # Check if username already exists
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute('SELECT * FROM users WHERE username = ?', (username,))
            user = c.fetchone()
            
            if user:
                error_message = "Username already exists!"
                return render_template('index.html', error_message=error_message)
            
            # Hash the password for secure storage
            hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
            
            # Print the hashed password for debugging
            print(f"Hashed password: {hashed_password}")

            # Insert the new user into the database
            c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_password))
            conn.commit()
            conn.close()

            success_message = "Account created successfully! You can now log in."
            return render_template('index.html', success_message=success_message)
    
    # If the user is logged in, show the home page; otherwise, show the login/signup page
    if 'logged_in' in session and session['logged_in']:
        return redirect(url_for('home'))
    
    return render_template('index.html', error_message=None, success_message=None)
    
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Fetch user from the database
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = c.fetchone()
        conn.close()

        # Check if user exists and the password is correct
        if user and check_password_hash(user[2], password):  # user[2] is the hashed password
            session['logged_in'] = True
            session['username'] = username  # Store username in session
            return redirect(url_for('home'))  # Redirect to the home page after login
        else:
            return render_template('index.html', error_message="Invalid credentials.")

    return render_template('index.html')


# New route for the signup page
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    return render_template('signup.html')  # Render the signup page

# Logout route
@app.route('/logout')
def logout():
    session.pop('logged_in', None)  # Clear the session
    session.pop('username', None)   # Clear the username from session
    return redirect(url_for('index'))  # Redirect to login page

@app.route('/result')
def result():
    # Retrieve the prediction index from session
    prediction_index = session.get('prediction_index', None)
    if prediction_index is None:
        return redirect(url_for('index'))  # If no prediction, redirect to home
    
    # Get plant info based on the prediction index
    plant_info = get_plant_info(prediction_index)
    
    return render_template('result.html', plant_info=plant_info)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/join_us')
def join_us():
    return render_template('join_us.html')

@app.route('/newsroom')
def newsroom():
    return render_template('newsroom.html')

@app.route('/ayurveda_and_you')
def ayurveda_and_you():
    return render_template('ayurveda_and_you.html')

# Update the upload route to handle non-plant images
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Get the predicted class index and confidence score
        predicted_class, predicted_confidence = predict_plant(file_path)
        
        if predicted_class is None:
            # If the prediction is None, show a message indicating this is not a plant
            return render_template('result.html', 
                                   message="This is not the image of a plant. Please upload the image of a plant.")
        
        # Convert the predicted class index (int64) to a regular Python int
        prediction_index = int(predicted_class)  # Ensure it's a regular int
        
        # Store prediction in session
        session['prediction_index'] = prediction_index
        
        # Get the plant info based on the predicted class index
        plant_info = get_plant_info(prediction_index)

        # Pass the file path and plant information to the result page
        return render_template('result.html', 
                               image_url=f"/{file_path}",  # Image URL for the uploaded file
                               plant_info=plant_info)  # Plant info to display

if __name__ == '__main__':
    app.run(debug=True)
