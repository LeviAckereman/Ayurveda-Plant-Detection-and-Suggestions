from flask import Blueprint, render_template, request, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3

# Define the blueprint for login and signup functionality
login_blueprint = Blueprint('login', __name__)

# Database setup function (create the users table if it doesn't exist)
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()

    # Create users table if it doesn't exist
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    username TEXT UNIQUE,
                    password TEXT
                )''')
    conn.commit()
    conn.close()

# Initialize the database
init_db()

# Route for Login & Signup combined
@login_blueprint.route('/', methods=['GET', 'POST'])
def index():
    # Set default credentials
    default_username = "user"
    default_password = "Adpas1114"
    
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        action = request.form.get('action')  # To differentiate between login and signup actions
        
        conn = sqlite3.connect('users.db')
        c = conn.cursor()

        if action == 'Login':
            # Login Logic
            # If the default credentials are used, allow the user to log in
            if username == default_username and password == default_password:
                session['logged_in'] = True
                session['username'] = default_username  # Store default username in session
                return redirect(url_for('upload'))  # Redirect to the upload page after login

            # Check if the user exists in the database
            c.execute('SELECT * FROM users WHERE username = ?', (username,))
            user = c.fetchone()

            if user:
                # Check password hash against stored password hash
                if check_password_hash(user[2], password):  # user[2] is the hashed password
                    session['logged_in'] = True
                    session['username'] = username  # Store username in session
                    return redirect(url_for('upload'))  # Redirect to upload page after login
                else:
                    error_message = "Invalid username or password!"
                    return render_template('index.html', error_message=error_message)
            else:
                error_message = "Invalid username or password!"
                return render_template('index.html', error_message=error_message)

        elif action == 'Signup':
            # Signup Logic
            c.execute('SELECT * FROM users WHERE username = ?', (username,))
            user = c.fetchone()

            if user:
                error_message = "Username already exists!"
                return render_template('index.html', error_message=error_message)

            # Hash the password for secure storage
            hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

            # Insert the new user into the database
            try:
                c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_password))
                conn.commit()  # Commit the changes to the database
                conn.close()
                # After signup, redirect to login page (not log them in automatically)
                return redirect(url_for('login.index'))  # Redirect to the login page after successful signup
            except Exception as e:
                # Catch any database errors
                error_message = f"Error while signing up: {str(e)}"
                return render_template('index.html', error_message=error_message)

    # If the user is logged in, show the upload page; otherwise, show the login page
    if 'logged_in' in session and session['logged_in']:
        return redirect(url_for('upload'))
    
    return render_template('index.html', error_message=None, success_message=None)

# Route for Logout
@login_blueprint.route('/logout')
def logout():
    session.pop('logged_in', None)  # Clear the session
    session.pop('username', None)   # Clear the username from session
    return redirect(url_for('home'))  # Redirect to home page after logout
