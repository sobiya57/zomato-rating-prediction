# 🍽️ Zomato Restaurant Rating Predictor

An end-to-end Machine Learning web application that predicts restaurant ratings based on customer behavior, pricing, and restaurant characteristics using Zomato data.

The project demonstrates the complete ML lifecycle — from data preprocessing and model training to deployment on Streamlit Cloud.

---
## 📸 App Screenshots

### 🏠 App Interface
![App UI](images/app_ui.png)

### ⭐ Prediction Output
![Prediction](images/prediction.png)

### 📊 Feature Importance
![Feature Importance](images/features_importance.png)

## 🚀 Live Demo

👉 **Streamlit App:**  
https://zomato-rating-prediction-kftvk7nl5zorwxrhygsyvn.streamlit.app/

---

## 📌 Project Overview

Restaurant ratings play a crucial role in customer decision-making.  
This project uses historical Zomato restaurant data to predict ratings using a supervised machine learning approach.

The trained model is hosted separately using **GitHub Releases** and dynamically loaded during runtime in the Streamlit app — making the deployment lightweight and scalable.

---

## 🧠 Machine Learning Approach

- **Problem Type:** Regression  
- **Target Variable:** Restaurant Rating  
- **Model Used:** RandomForest Regressor  
- **Why RandomForest?**
  - Handles non-linear relationships well
  - Robust to outliers
  - Provides feature importance for explainability

---

## 📊 Features Used for Prediction

- Online Order Availability
- Table Booking Availability
- Restaurant Location
- Restaurant Type
- Cuisines
- Approximate Cost for Two (₹)
- Number of Customer Votes

---

## 📈 Model Explainability

The application provides **feature importance visualization** to explain:
- Why a particular rating was predicted
- Which features influenced the prediction the most

This improves transparency and trust in the ML model.

---

## 🛠 Tech Stack

- **Programming Language:** Python  
- **Data Analysis:** Pandas, NumPy  
- **Machine Learning:** Scikit-learn  
- **Visualization:** Matplotlib  
- **Web App Framework:** Streamlit  
- **Model Hosting:** GitHub Releases  
- **Deployment:** Streamlit Cloud  
- **Version Control:** Git & GitHub  

---

## 📂 Project Structure

zomato-rating-prediction/
│
├── app/
│ └── app.py # Streamlit application
│
├── data/
│ └── zomato.csv # Dataset (local training)
│
├── models/
│ └── zomato_rating_model.joblib # Trained model (ignored in git)
│
├── notebooks/
│ └── 01_data_cleaning.ipynb # Data cleaning & model training
│
├── requirements.txt
├── .gitignore
└── README.md


---

## ⚙️ Model Hosting Strategy

- The trained ML model is **NOT committed to GitHub**
- Instead, it is uploaded as a **GitHub Release asset**
- The Streamlit app downloads the model dynamically at runtime

✔ Prevents large file issues  
✔ Keeps repository clean  
✔ Production-friendly deployment approach  

---

## ▶️ How to Run Locally

1️⃣ Clone the repository
```bash
git clone https://github.com/sobiya57/zomato-rating-prediction.git
cd zomato-rating-prediction

2️⃣ Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run Streamlit app
streamlit run app/app.py


## 📊 Dataset

- The dataset is based on Zomato restaurant listings.
- It contains information such as restaurant type, location, cuisines, pricing, online ordering, table booking, votes, and ratings.
- The dataset was cleaned and preprocessed before training the machine learning model.
- This dataset is used strictly for educational and project demonstration purposes.


## 📌 Project Highlights

- End-to-end Machine Learning project
- Real-world restaurant rating prediction use case
- Data cleaning and preprocessing using Pandas
- Feature engineering and categorical encoding
- RandomForest regression model for prediction
- Feature importance visualization for explainability
- Dynamic model loading using GitHub Releases
- Lightweight and scalable Streamlit deployment
- Clean project structure following industry standards
- Resume-ready and interview-ready project


## 🧾 Disclaimer

This project is developed for learning and demonstration purposes only.  
It is not affiliated with, sponsored by, or endorsed by Zomato.


## 👩‍💻 Author

**Sobiya Begum**  
Aspiring Data Scientist | Machine Learning Enthusiast  

🔗 GitHub: https://github.com/sobiya57
