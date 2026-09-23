
# 💳 Credit Card Fraud Detection

A Machine Learning based application that detects potentially fraudulent credit card transactions using Logistic Regression and provides prediction results through an interactive Streamlit web interface.

---

## 📌 Project Overview

Credit card fraud is a major financial problem where unauthorized transactions can cause significant losses.

This project uses Machine Learning to classify credit card transactions into:

- ✅ Legitimate Transaction
- 🚨 Fraudulent Transaction

The trained Machine Learning model is integrated with a Streamlit application where users can enter transaction details and receive a prediction along with the probability of legitimate and fraudulent transactions.

---

## 🎯 Problem Statement

Credit card transaction datasets are highly imbalanced because fraudulent transactions are much fewer than legitimate transactions.

The goal of this project is to build a Machine Learning model that can identify fraudulent transactions while paying particular attention to detecting the minority fraud class.

---

## 💡 Solution

The project follows this workflow:

Credit Card Transaction Dataset
        ↓
Data Preprocessing
        ↓
Feature Preparation
        ↓
Train-Test Split
        ↓
Logistic Regression
        ↓
Model Evaluation
        ↓
Save Trained Model
        ↓
Streamlit Web App
        ↓
Fraud/Legitimate Prediction

---

## 📊 Dataset

The project uses the Credit Card Fraud Detection dataset.

### Dataset Statistics

| Property | Value |
|---|---:|
| Total Transactions | 284,807 |
| Features | 30 |
| Legitimate Transactions | 284,315 |
| Fraudulent Transactions | 492 |
| Missing Values | 0 |

The dataset contains:

- Time
- V1 to V28
- Amount
- Class

The Class column is the target variable:

0 → Legitimate  
1 → Fraudulent

---

## 🤖 Machine Learning Model

### Logistic Regression

Logistic Regression is used as the classification algorithm.

The model pipeline includes:

1. Feature scaling using StandardScaler
2. Logistic Regression
3. Class-weight balancing

The model uses:

class_weight="balanced"

This helps give more importance to the minority fraud class.

---

## 📈 Model Performance

The model was evaluated on a separate test dataset.

| Metric | Score |
|---|---:|
| Accuracy | 97.55% |
| Precision | 6.10% |
| Recall | 91.84% |
| F1-Score | 11.44% |

### Fraud Class Performance

Precision: 6.10%  
Recall: 91.84%  
F1-Score: 11.44%

The fraud-class recall of 91.84% means the model identified 90 out of 98 fraud cases in the test set.

Because the dataset is highly imbalanced, accuracy alone does not fully describe fraud-detection performance. Precision, recall and F1-score are also considered.

---

## 🔎 Confusion Matrix

The model produced the following confusion matrix on the test set:

[[55478  1386]
 [    8    90]]

### Interpretation

- True Negatives = 55,478
- False Positives = 1,386
- False Negatives = 8
- True Positives = 90

---

## 🖥️ Streamlit Application

The project includes an interactive Streamlit web application.

Users can enter:

- ⏱️ Transaction Time
- 💰 Transaction Amount
- 🧮 V1 to V28 transaction features

After clicking Predict Transaction, the application displays:

- ✅ Legitimate/Fraud prediction
- 📊 Legitimate probability
- 🚨 Fraud probability
- 📈 Prediction probability chart
- 🔎 Entered transaction details

---

## 🛠️ Technologies Used

### Programming Language

- Python

### Machine Learning

- Scikit-learn
- Logistic Regression
- StandardScaler

### Data Processing

- Pandas
- NumPy

### Model Saving

- Joblib

### Web Application

- Streamlit

### Visualization

- Matplotlib
- Seaborn

### Version Control

- Git
- GitHub

---

## 📁 Project Structure

Credit-Card-Fraud-Detection/
│
├── app.py
├── train_model.py
├── Fraud Detection Model.pkl
├── requirements.txt
├── README.md
├── .gitignore
│
└── data/
    └── creditcard.csv

Note: The dataset is not included in the GitHub repository because its size exceeds GitHub's standard individual file size limit. Download the dataset separately and place it inside the data folder.

---

## ⚙️ Installation

### 1. Clone the Repository

git clone https://github.com/jayaMishra01/Credit-Card-Fraud-Detection.git

### 2. Open the Project Directory

cd Credit-Card-Fraud-Detection

### 3. Install Required Libraries

pip install -r requirements.txt

---

## 📂 Dataset Setup

Create a folder named:

data

Place the dataset file inside it:

data/creditcard.csv

---

## 🧠 Train the Model

If you want to train the model again, run:

python train_model.py

This will train the Logistic Regression model and generate:

Fraud Detection Model.pkl

---

## 🚀 Run the Streamlit Application

Run:

streamlit run app.py

The application will open in your browser.

---

## 🔮 Future Improvements

Possible improvements include:

- Random Forest and XGBoost model comparison
- Hyperparameter tuning
- Threshold optimization
- Reduction of false positives
- ROC-AUC and PR-AUC evaluation
- Feature importance analysis
- Model comparison
- Cloud deployment
- Real-time transaction monitoring
- Improved UI and visualization

---

## 📚 Learning Outcomes

Through this project, I learned:

- Data preprocessing
- Exploratory data analysis
- Classification algorithms
- Logistic Regression
- Feature scaling
- Handling imbalanced datasets
- Train-test splitting
- Model evaluation
- Precision, Recall and F1-score
- Confusion matrix
- Model serialization using Joblib
- Streamlit application development
- Git and GitHub project management

---

## ⚠️ Disclaimer

This project is created for educational and demonstration purposes only.

It should not be used as a production financial fraud detection system or relied upon for real-world financial decisions.

---

## 👩‍💻 Author

Jaya

Computer Science & Engineering Student

### GitHub

https://github.com/JayaMishra01

### LinkedIn

https://linkedin.com/in/jaya-1569033a

---

## ⭐ Project Highlights

- Machine Learning based fraud detection
- Handles highly imbalanced transaction data
- 91.84% fraud-class recall
- Interactive Streamlit interface
- Prediction probability visualization
- Complete GitHub project structure
  
