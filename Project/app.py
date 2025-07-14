import os
import pickle
import cv2
import numpy as np
import mediapipe as mp
import mysql.connector
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash
from flask_bcrypt import Bcrypt
import re
import dns.resolver
from werkzeug.utils import secure_filename
from scipy.signal import butter, filtfilt
import scipy.fftpack

# Initialize Flask App and Extensions
app = Flask(__name__)
bcrypt = Bcrypt(app)
app.secret_key = os.getenv('SECRET_KEY', 'your_secret_key')  # Use environment variables in production

# Allowed file extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'mp4'}

# Database Connection
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="1234",
        database="deepfake_db"
    )

# Get user by email
def get_user_by_email(email):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
    user = cursor.fetchone()
    conn.close()
    return user

# Get user by ID
def get_user_by_id(user_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
    user = cursor.fetchone()
    conn.close()
    return user

# Create user
def create_user(username, email, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    cursor.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)", 
                   (username, email, hashed_password))
    conn.commit()
    conn.close()

# Load deepfake detection model
model_path = "deepfake_model.pkl"
if os.path.exists(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
else:
    raise FileNotFoundError("❌ Model file not found!!")

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

# Email Validation
ALLOWED_DOMAINS = {
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "icloud.com",
    "live.com", "aol.com", "protonmail.com", "zoho.com", "msn.com", "edu"
}

def is_valid_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        return False, "Invalid email format."
    domain = email.split('@')[-1]
    if domain in ALLOWED_DOMAINS or domain.endswith('.edu'):
        return True, None
    try:
        dns.resolver.resolve(domain, 'MX')
        return True, None
    except Exception as e:
        return False, f"❌ Domain error: {str(e)}"

# Butterworth filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# Extract facial regions
def extract_face_regions(frame, face_landmarks):
    h, w, _ = frame.shape
    x_min = int(face_landmarks.location_data.relative_bounding_box.xmin * w)
    y_min = int(face_landmarks.location_data.relative_bounding_box.ymin * h)
    box_width = int(face_landmarks.location_data.relative_bounding_box.width * w)
    box_height = int(face_landmarks.location_data.relative_bounding_box.height * h)

    forehead = frame[y_min:y_min + int(0.2 * box_height), x_min + int(0.3 * box_width): x_min + int(0.7 * box_width)]
    left_cheek = frame[y_min + int(0.4 * box_height): y_min + int(0.6 * box_height), x_min: x_min + int(0.2 * box_width)]
    right_cheek = frame[y_min + int(0.4 * box_height): y_min + int(0.6 * box_height), x_min + int(0.8 * box_width): x_min + box_width]
    nose = frame[y_min + int(0.3 * box_height): y_min + int(0.5 * box_height), x_min + int(0.4 * box_width): x_min + int(0.6 * box_width)]

    return [forehead, left_cheek, right_cheek, nose]

# Estimate heart rate (for videos)
def estimate_heart_rate(video_path, duration=10):
    cap = cv2.VideoCapture(video_path)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    num_frames = duration * frame_rate

    intensity_values = []
    frame_count = 0

    while cap.isOpened() and frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                face_regions = extract_face_regions(frame, detection)
                avg_intensity = np.mean([np.mean(region[:, :, 1]) for region in face_regions if region.size > 0])
                intensity_values.append(avg_intensity)

        frame_count += 1

    cap.release()

    if len(intensity_values) < 2:
        return None, None

    filtered_signal = butter_bandpass_filter(intensity_values, 0.7, 4.0, frame_rate, order=3)
    fft_result = np.abs(scipy.fftpack.fft(filtered_signal))
    freqs = scipy.fftpack.fftfreq(len(filtered_signal), d=1/frame_rate)
    peak_index = np.argmax(fft_result[freqs > 0])
    heart_rate = freqs[freqs > 0][peak_index] * 60
    std_dev = np.std(filtered_signal)

    return heart_rate, std_dev

# Check blurriness (for videos and images)
def check_blurriness(file_path, is_video=True):
    if is_video:
        cap = cv2.VideoCapture(file_path)
        total_frames, blurry_count = 0, 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            total_frames += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < 50:
                blurry_count += 1
        cap.release()
        return (blurry_count / total_frames) * 100 if total_frames > 0 else 0
    else:
        img = cv2.imread(file_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return 100 if laplacian_var < 50 else 0

# Detect deepfake (for videos and images)
def detect_deepfake(file_path, is_video=True):
    blur_score = check_blurriness(file_path, is_video)
    if is_video:
        heart_rate, hr_variation = estimate_heart_rate(file_path, duration=10)
        deepfake_score = 0
        if heart_rate:
            deepfake_score += abs(heart_rate - 72) * 1.8
            deepfake_score += hr_variation * 0.6
        deepfake_score += (blur_score * 0.4)
    else:
        # For images, rely on blurriness and model prediction
        img = cv2.imread(file_path)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_img)
        deepfake_score = blur_score * 0.4
        if results.detections:
            deepfake_score += 20  # Placeholder adjustment for face detection in images

    confidence = min(max(100 - deepfake_score, 0), 100)
    return confidence

# Check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Routes
@app.route("/")
def index():
    user_initial = None
    if 'loggedin' in session:
        user_initial = session['username'][0].upper()
    return render_template("index.html", user_initial=user_initial)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        is_valid, error_message = is_valid_email(email)
        if not is_valid:
            flash(error_message, 'danger')
            return redirect(url_for('signup'))
        create_user(username, email, password)
        flash('✅ Account created successfully! Please log in.', 'success')
        return redirect(url_for('signin'))
    return render_template('signup.html')

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        email = request.form['email']
        password_candidate = request.form['password']
        user = get_user_by_email(email)
        if user and bcrypt.check_password_hash(user['password'], password_candidate):
            session['loggedin'] = True
            session['id'] = user['id']
            session['username'] = user['username']
            session['email'] = user['email']
            session['user_initial'] = user['username'][0].upper()
            flash('✅ Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('❌ Incorrect email or password.', 'danger')
    return render_template('signin.html')

@app.route('/profile')
def profile():
    if 'loggedin' not in session:
        flash("❌ Please log in to access your profile.", "danger")
        return redirect(url_for('signin'))
    
    user_initial = session['username'][0].upper()
    return render_template('profile.html', 
                           username=session['username'], 
                           email=session['email'], 
                           user_initial=user_initial)

@app.route('/premium')
def premium():
    return render_template('premium.html')

@app.route('/delete-account', methods=['POST'])
def delete_account():
    if 'loggedin' not in session:
        flash("❌ You need to log in first!", "danger")
        return redirect(url_for('signin'))

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users WHERE id = %s", (session['id'],))
    conn.commit()
    conn.close()

    session.clear()
    flash("🗑️ Account deleted permanently!", "success")
    return redirect(url_for('signin'))

@app.route('/logout', methods=['GET', 'POST'])
def logout():
    session.clear()
    flash("✅ Signed out successfully!", "success")
    return redirect(url_for('signin'))

@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        message = request.form["message"]

        db = get_db_connection()
        cursor = db.cursor()
        query = "INSERT INTO contact_messages (name, email, message) VALUES (%s, %s, %s)"
        cursor.execute(query, (name, email, message))
        db.commit()
        cursor.close()
        db.close()

        return jsonify({"message": "Message received successfully!"}), 200
    
    return render_template("contact.html")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash("❌ No file uploaded!", "danger")
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        flash("❌ No file selected!", "danger")
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join("static/uploads", filename)
        os.makedirs("static/uploads", exist_ok=True)
        file.save(file_path)

        # Determine if it's a video or an image
        is_video = filename.lower().endswith('.mp4')
        deepfake_score = detect_deepfake(file_path, is_video)
        result = "Fake" if deepfake_score >= 60 else "Real"

        # URL to access the uploaded file
        file_url = url_for('static', filename=f'uploads/{filename}', _external=True)

        return render_template("result.html", score=round(deepfake_score, 2), result=result, is_video=is_video, file_url=file_url)

    else:
        flash("❌ Only JPG, JPEG, PNG, and MP4 files are allowed!", "danger")
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
