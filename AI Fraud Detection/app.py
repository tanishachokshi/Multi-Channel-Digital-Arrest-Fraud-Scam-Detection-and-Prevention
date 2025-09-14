from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import os
import joblib
import re
from datetime import datetime
import json
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
# Strong secret key for session encryption
app.secret_key = os.urandom(24)

# Database initialization
def init_db():
    """Initialize the SQLite database with required tables"""
    conn = sqlite3.connect('fraud_detection.db')
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            email TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Detected scams table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detected_scams (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            text_content TEXT NOT NULL,
            is_scam BOOLEAN NOT NULL,
            confidence REAL NOT NULL,
            risk_level TEXT NOT NULL,
            indicators TEXT,
            keywords_found TEXT,
            explanation TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Reported scams table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reported_scams (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            message_type TEXT NOT NULL,
            content TEXT NOT NULL,
            source TEXT,
            description TEXT,
            severity TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()

# Initialize database on startup
init_db()

# Load ML models
try:
    scam_classifier = joblib.load('models/scam_classifier.pkl')
    vectorizer = joblib.load('models/vectorizer.pkl')

    models_loaded = True
    print("✓ ML models loaded successfully")
except FileNotFoundError:
    models_loaded = False
    scam_classifier = None
    vectorizer = None
    print("⚠ Warning: ML model files not found. Detection features will use pattern matching only.")
except Exception as e:
    models_loaded = False
    scam_classifier = None
    vectorizer = None
    print(f"⚠ Warning: Error loading ML models ({str(e)}). Detection features will use pattern matching only.")

# ---------- Home Page ----------
@app.route("/")
def home():
    return render_template("home.html")


# ---------- Auth (Login/Register) ----------
@app.route("/auth/<action>", methods=["GET", "POST"])
def auth(action):
    if action not in ["login", "register"]:
        flash("Invalid action!", "danger")
        return redirect(url_for("home"))

    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        # REGISTER
        if action == "register":
            conn = sqlite3.connect('fraud_detection.db')
            cursor = conn.cursor()
            
            # Check if username already exists
            cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
            if cursor.fetchone():
                flash("Username already exists!", "danger")
            else:
                password_hash = generate_password_hash(password)
                cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", 
                             (username, password_hash))
                conn.commit()
                conn.close()
                flash("Registration successful! Login now.", "success")
                return redirect(url_for("auth", action="login"))

        # LOGIN
        elif action == "login":
            conn = sqlite3.connect('fraud_detection.db')
            cursor = conn.cursor()
            
            cursor.execute("SELECT id, password_hash FROM users WHERE username = ?", (username,))
            user = cursor.fetchone()
            conn.close()
            
            if user and check_password_hash(user[1], password):
                session["user_id"] = user[0]
                session["username"] = username
                flash(f"Welcome, {username}!", "success")
                return redirect(url_for("dashboard"))
            else:
                flash("Invalid username or password!", "danger")

    return render_template("auth.html", action=action)


# ---------- Dashboard ----------
@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        flash("Please login first!", "warning")
        return redirect(url_for("auth", action="login"))

    conn = sqlite3.connect('fraud_detection.db')
    cursor = conn.cursor()
    
    # Get recent detections
    cursor.execute('''
        SELECT text_content, is_scam, confidence, risk_level, indicators, keywords_found, explanation, timestamp
        FROM detected_scams 
        WHERE user_id = ? 
        ORDER BY timestamp DESC 
        LIMIT 5
    ''', (session["user_id"],))
    recent_detections = cursor.fetchall()
    
    # Get statistics
    cursor.execute("SELECT COUNT(*) FROM detected_scams WHERE user_id = ?", (session["user_id"],))
    total_detections = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM reported_scams WHERE user_id = ?", (session["user_id"],))
    total_reports = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM detected_scams WHERE user_id = ? AND risk_level = 'high'", (session["user_id"],))
    high_risk_detections = cursor.fetchone()[0]
    
    conn.close()
    
    stats = {
        'total_detections': total_detections,
        'total_reports': total_reports,
        'high_risk_detections': high_risk_detections,
        'models_loaded': models_loaded
    }
    
    return render_template("dashboard.html", 
                         username=session["username"], 
                         recent_detections=recent_detections,
                         stats=stats)

# ---------- Scam Detection ----------
@app.route("/detect", methods=["GET", "POST"])
def detect_scam():
    if "user_id" not in session:
        flash("Please login first!", "warning")
        return redirect(url_for("auth", action="login"))
    
    if request.method == "POST":
        text_input = request.form.get("text_input", "").strip()
        
        if not text_input:
            flash("Please enter some text to analyze!", "warning")
            return render_template("detect.html", username=session["username"])
        
        # Analyze the text
        result = analyze_text(text_input)
        
        # Store detection result in database
        conn = sqlite3.connect('fraud_detection.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO detected_scams 
            (user_id, text_content, is_scam, confidence, risk_level, indicators, keywords_found, explanation)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            session["user_id"],
            text_input,
            result['is_scam'],
            result['confidence'],
            result['risk_level'],
            json.dumps(result['indicators']),
            json.dumps(result['keywords_found']),
            result['explanation']
        ))
        conn.commit()
        conn.close()
        
        return render_template("detect.html", 
                             username=session["username"], 
                             result=result,
                             analyzed_text=text_input)
    
    return render_template("detect.html", username=session["username"])

# ---------- Manual Reporting ----------
@app.route("/report", methods=["GET", "POST"])
def report_scam():
    if "user_id" not in session:
        flash("Please login first!", "warning")
        return redirect(url_for("auth", action="login"))
    
    if request.method == "POST":
        conn = sqlite3.connect('fraud_detection.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO reported_scams 
            (user_id, message_type, content, source, description, severity)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            session["user_id"],
            request.form.get("message_type"),
            request.form.get("content", "").strip(),
            request.form.get("source", "").strip(),
            request.form.get("description", "").strip(),
            request.form.get("severity", "medium")
        ))
        conn.commit()
        conn.close()
        
        flash("Thank you for your report! It will help improve our detection system.", "success")
        return redirect(url_for("report_scam"))
    
    return render_template("report.html", username=session["username"])

# ---------- Educational Resources ----------
@app.route("/education")
def education():
    if "user_id" not in session:
        flash("Please login first!", "warning")
        return redirect(url_for("auth", action="login"))
    
    return render_template("education.html", username=session["username"])

# ---------- API Endpoints ----------
@app.route("/api/detect", methods=["POST"])
def api_detect():
    data = request.get_json()
    text = data.get("text", "").strip()
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    result = analyze_text(text)
    return jsonify(result)

@app.route("/api/stats")
def api_stats():
    if "username" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    stats = {
        'total_detections': len(detected_scams),
        'total_reports': len(reported_scams),
        'high_risk_detections': len([s for s in detected_scams if s.get('risk_level') == 'high']),
        'models_loaded': models_loaded
    }
    return jsonify(stats)

# ---------- Helper Functions ----------
def analyze_text(text):
    """Analyze text for scam indicators using ML model and pattern matching"""
    global models_loaded
    
    result = {
        'is_scam': False,
        'confidence': 0.0,
        'risk_level': 'low',
        'indicators': [],
        'explanation': '',
        'keywords_found': []
    }
    
    # Use ML model if available
    if models_loaded and scam_classifier is not None and vectorizer is not None:
        try:
            text_vectorized = vectorizer.transform([text])
            prediction = scam_classifier.predict(text_vectorized)[0]
            probability = scam_classifier.predict_proba(text_vectorized)[0]
            
            result['is_scam'] = bool(prediction)
            result['confidence'] = float(max(probability))
            
            if result['confidence'] > 0.8:
                result['risk_level'] = 'high'
            elif result['confidence'] > 0.6:
                result['risk_level'] = 'medium'
            else:
                result['risk_level'] = 'low'
                
        except Exception as e:
            print(f"ML model error: {e}")
            # Fall back to pattern-based detection
            models_loaded = False
    
    # Pattern-based detection for additional indicators
    scam_patterns = {
        'urgent_action': [r'\b(urgent|immediately|act now|limited time|expires soon)\b', 'Urgent action required'],
        'financial_gain': [r'\b(free money|guaranteed profit|investment opportunity|get rich quick)\b', 'Promises of financial gain'],
        'authority_impersonation': [r'\b(IRS|police|court|government|official)\b', 'Authority impersonation'],
        'personal_info': [r'\b(SSN|social security|bank account|credit card|password)\b', 'Requests for personal information'],
        'threats': [r'\b(arrest|warrant|legal action|suspended|blocked)\b', 'Threats or intimidation'],
        'suspicious_links': [r'https?://[^\s]+', 'Suspicious links'],
        'phone_numbers': [r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', 'Phone numbers'],
        'email_addresses': [r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'Email addresses']
    }
    
    found_keywords = []
    for pattern_name, (pattern, description) in scam_patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            result['indicators'].append(description)
            found_keywords.extend(matches)
    
    result['keywords_found'] = list(set(found_keywords))
    
    # If no ML model was used, determine scam status based on indicators
    if not models_loaded or result['confidence'] == 0.0:
        if len(result['indicators']) >= 3:
            result['is_scam'] = True
            result['confidence'] = 0.8
            result['risk_level'] = 'high'
        elif len(result['indicators']) >= 2:
            result['is_scam'] = True
            result['confidence'] = 0.6
            result['risk_level'] = 'medium'
        elif len(result['indicators']) >= 1:
            result['confidence'] = 0.4
            result['risk_level'] = 'low'
        else:
            result['confidence'] = 0.1
            result['risk_level'] = 'low'

    # Generate explanation
    if result['is_scam'] or result['indicators']:
        if result['is_scam']:
            result['explanation'] = f"This message appears to be a scam with {result['confidence']:.1%} confidence. "
        else:
            result['explanation'] = "This message shows some suspicious characteristics. "
        
        if result['indicators']:
            result['explanation'] += f"Detected indicators: {', '.join(result['indicators'])}. "
        
        result['explanation'] += "Be cautious and verify the source before taking any action."
    else:
        result['explanation'] = "This message appears to be legitimate. No obvious scam indicators detected."
    
    return result


# ---------- Logout ----------
@app.route("/logout")
def logout():
    session.pop("user_id", None)
    session.pop("username", None)
    flash("Logged out successfully!", "success")
    return redirect(url_for("home"))


# ---------- Run the app ----------
if __name__ == "__main__":
    app.run(debug=True)