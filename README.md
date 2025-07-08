# Deepfake Detection Web App

A web application that detects whether uploaded images or videos are real or deepfakes using a custom-trained machine learning model based on **thermal** and **heart rate** analysis.


## 💡 What It Does

- Upload an image or video  
- Get a prediction: **Real or Fake**  
- Uses a `.pkl` ML model trained with:
  - Thermal pattern analysis  
  - Heartbeat signal detection

Developed with **Flask**, styled with **HTML/CSS/JavaScript**, and powered by machine learning.


## 🧠 How I Made It

This project was built by leveraging my Python fundamentals, combined with strong **prompt engineering skills** using tools like **ChatGPT** and **Grok**. Every part — from training the ML model to connecting it with the web app — was guided by smart AI prompting and research.

Model training was done on **Google Colab**, and integration was done through **VS Code** using Flask.


## 🔧 Tech Stack

- Python, Flask  
- HTML, CSS, JavaScript  
- Machine Learning (OpenCV, NumPy, Scikit-learn)  
- Git & GitHub


## 🗂️ Project Structure

deepfake-detection/
├── templates/ # HTML files
├── static/ # CSS, JS, images
├── app.py # Main Flask app
├── test.py # ML testing script
├── deepfake_model.pkl # Trained model
├── requirements.txt # Dependencies
├── .gitignore
└── LICENSE


## 🚀 Getting Started

1. Clone the repo:
   bash:
      1. git clone https://github.com/lubaanah/deepfake-detection
cd deepfake-detection

    2. python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

    3. python app.py

📜 License:

Licensed under the MIT License. Use it, learn from it, remix it — just give credit where it’s due.

📝 Future Scope
Deploy to a live server (Render/Streamlit/Replit)

Improve model accuracy

Add real-time webcam deepfake detection

User authentication system


🤝 Acknowledgements
Built with the support of AI tools: ChatGPT & Grok
Thanks to open-source tools, documentation, and the ML/dev community

Project by lubaanah

@lubaanah
