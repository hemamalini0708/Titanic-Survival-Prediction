# 🚢 Titanic Survival Prediction System

An end-to-end Machine Learning project that predicts whether a passenger **Survived** or **Did Not Survive** the Titanic disaster based on demographic and travel attributes.

This project follows a structured ML pipeline including data preprocessing, feature encoding, model training, evaluation, and deployment using Streamlit.

---

## Problem Statement

The objective of this project is to build a classification model that predicts passenger survival using historical Titanic dataset.

The system classifies passengers into:

- **Survived (1)**
- **Did Not Survive (0)**

This helps understand key survival factors such as gender, age, fare, and port of embarkation.

---

##  Dataset

The dataset contains passenger demographic and travel information including:

- Age  
- Fare  
- Parents/Children Aboard (Parch)  
- Gender  
- Port of Embarkation  

###  Target Variable

- **1 → Survived**
- **0 → Did Not Survive**

---

##  Machine Learning Pipeline

The project follows a structured ML workflow:

### 1. Data Preprocessing

- Handled missing values  
- Encoded categorical features  
- Structured numerical and categorical inputs  
- Prepared final feature matrix  

---

### 2. Feature Engineering

- One-Hot Encoding for:
  - Gender  
  - Port of Embarkation  
- Combined encoded and numerical features  

---

### 3. Model Training

Model implemented:

- **Naive Bayes (GaussianNB)**  

The model predicts both survival class and probability confidence score.

---

## 📈 Model Evaluation

Models were evaluated using:

- Accuracy Score  
- Confusion Matrix  
- Classification Report  
- ROC-AUC Curve comparison  

The ROC curve was used to compare model performance across:
- Naive Bayes (NB)
- Logistic Regression (LR)
- Decision Tree (DT)

The best-performing model was selected based on comparative analysis.

<img width="730" height="291" alt="image" src="https://github.com/user-attachments/assets/4f1573a1-d699-4ab3-8880-f7b62470b374" />


---

## 🚀 Deployment

The trained model is deployed using **Streamlit**.

The web application allows users to:

- Enter passenger details  
- Automatically process and encode inputs  
- Predict survival outcome  
- Display probability confidence  

---

## ▶ Run Locally

```bash
pip install -r requirements.txt
streamlit run Titanic_streamlit.py
```

**Tech Stack**

- Python

- Pandas

- NumPy

- Scikit-Learn

- Streamlit

**Project Structure**
```
titanic-survival-prediction/
│
├── data/
├── model/
├── src/
├── Titanic_streamlit.py
├── README.md
├── requirements.txt
└── .gitignore
```

**Analytical Impact**
- Demonstrates end-to-end ML workflow
- Implements probabilistic classification
- Applies feature encoding techniques
- Deploys ML model using Streamlit

**Author**

Hema Malini Gangumalla

Aspiring Data Scientist

📧 hemamalinig07@gmail.com

**License**

MIT License
