🍽️ Zomato Restaurant Rating Prediction

An end-to-end Machine Learning project that predicts restaurant ratings based on services, location, cuisine, cost, and customer engagement data.
The project includes data cleaning, feature engineering, model training, and deployment using Streamlit.


📌 Project Overview

Restaurant ratings play a crucial role in customer decision-making on food delivery platforms.
This project aims to predict restaurant ratings using historical data from Zomato, helping understand which factors influence customer ratings the most.


🎯 Problem Statement

Given restaurant attributes such as:

Online ordering availability

Table booking option

Location

Restaurant type

Cuisines

Approximate cost for two

Number of votes

➡️ Predict the restaurant rating (out of 5) using machine learning.


🧠 Machine Learning Approach

Type: Regression Problem

Target Variable: rate

Model Used: Random Forest Regressor

Evaluation Metrics:

Mean Absolute Error (MAE)

R² Score

Random Forest was chosen because it:

Handles non-linear relationships well

Works effectively with mixed data types

Reduces overfitting compared to single models


zomato-rating-prediction/
│
├── app/
│   └── app.py                  # Streamlit application
│
├── data/
│   └── zomato.csv               # Dataset (not pushed to GitHub)
│
├── models/
│   └── zomato_rating_model.joblib
│
├── notebooks/
│   └── 01_data_cleaning.ipynb   # Data cleaning & model training
│
├── requirements.txt
├── README.md
└── .gitignore


🔍 Data Preprocessing

Key preprocessing steps:

Removed irrelevant columns

Cleaned rating values (4.1/5, NEW, -)

Converted ratings to numeric format

Handled missing values

Encoded categorical variables using OneHotEncoding

Built a preprocessing + model pipeline


⚙️ Tech Stack

Programming Language: Python

Libraries:

pandas

numpy

scikit-learn

joblib

streamlit

IDE: VS Code

Deployment: Streamlit Cloud


🚀 How to Run the Project Locally

1️⃣ Clone the repository

git clone https://github.com/sobiya57/zomato-rating-prediction.git
cd zomato-rating-prediction

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

streamlit run app/app.py


🖥️ Streamlit App Features

User-friendly UI

Accepts restaurant details as input

Predicts restaurant rating instantly

Displays rating clearly out of 5


📈 Results

Achieved strong predictive performance with Random Forest

Model generalizes well on unseen data

Handles categorical features effectively


📌 Future Improvements

Dropdowns instead of text inputs

Feature importance visualization

Location-based analysis

Model comparison (Linear vs Random Forest)

UI enhancements


💼 Resume Value

This project demonstrates:

Real-world data cleaning

Feature engineering

Model building & evaluation

Pipeline usage

Model deployment

End-to-end ML workflow


🙌 Author

Sobiya Begum
Aspiring Data Scientist | Machine Learning | Data Analysis


⭐ Acknowledgements

Dataset inspired by Zomato restaurant data for educational purposes.



