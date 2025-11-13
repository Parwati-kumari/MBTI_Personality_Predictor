# 🧠 MBTI Personality Predictor

This project is a **Streamlit web application** that predicts your **MBTI (Myers-Briggs Type Indicator)** personality type based on the text you write.  
It uses a trained machine learning model to analyze your writing style and infer your most likely MBTI type (e.g., INTJ, ENFP, ISFJ, etc.).

---

## 🚀 Project Overview

The **MBTI Personality Predictor** helps users explore their potential personality type by entering a short self-description or paragraph about themselves.  
The app processes the text, cleans it, extracts meaningful features using **TF-IDF vectorization**, and then predicts the MBTI type using a trained classifier.

### 🧩 Features
- Predicts your personality type from any input text  
- Displays **top 3 predicted MBTI types** with confidence scores  
- Shows interactive **bar charts** for prediction confidence  
- Beautiful **Streamlit UI** with animations and hover effects  
- Includes information cards for **all 16 MBTI personality types**

---

## 🧠 Technologies Used

- **Python 3.10+**
- **Streamlit** – for building the web app  
- **Scikit-learn** – for model training and TF-IDF vectorization  
- **Joblib** – for saving and loading the model  
- **NLTK** – for text preprocessing and stopword removal  
- **Matplotlib** – for visualization  
- **Requests & Streamlit-Lottie** – for animations and external assets  

---

## 📁 Project Structure
```
Mbti_project/
│
├── frontend/
│ └── app.py # Streamlit web app
│
├── backend/
│ ├── personality_model.pkl # Trained ML model
│ ├── tfidf_vectorizer.pkl # TF-IDF vectorizer
│ └── label_encoder.pkl # Label encoder
│
├── requirements.txt # Project dependencies
└── README.md # Project documentation
```
---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/MBTI_Personality_Predictor.git
cd MBTI_Personality_Predictor
```
### 2️⃣ Install Dependencies
```
pip install -r requirements.txt
```
### 3️⃣ Run the Application
```
streamlit run frontend/app.py
```

Then open the local URL displayed in your terminal (usually http://localhost:8501).
