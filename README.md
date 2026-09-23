
💳 Credit Card Fraud Detection
A Machine Learning project that detects potentially fraudulent credit card transactions using Python, Scikit-learn, and Logistic Regression.
The project addresses the highly imbalanced nature of fraud detection using class balancing and evaluates the model using Accuracy, Precision, Recall, F1-Score, and a Confusion Matrix.
📌 Project Overview
Credit card fraud is a major challenge in digital payments. Fraudulent transactions are extremely rare compared to legitimate transactions, making fraud detection a challenging imbalanced classification problem.
This project develops a Machine Learning model that analyzes transaction features and predicts whether a transaction is:
0 → Legitimate Transaction
1 → Fraudulent Transaction
The trained model is also integrated with a Streamlit web application for interactive transaction prediction.
🎯 Problem Statement
The objective of this project is to develop a Machine Learning system that can identify fraudulent credit card transactions while reducing the number of missed fraudulent transactions.
Because fraudulent transactions represent only a small percentage of the dataset, the model needs to handle class imbalance effectively.
💡 Solution
The project follows a complete Machine Learning pipeline:
Load the credit card transaction dataset.
Check the dataset structure and missing values.
Analyze the distribution of legitimate and fraudulent transactions.
Separate features and target variable.
Split the data into training and testing datasets.
Apply feature scaling using StandardScaler.
Train a Logistic Regression model.
Handle class imbalance using class_weight="balanced".
Generate predictions on the test dataset.
Evaluate the model using multiple classification metrics.
Save the trained model using Joblib.
Use the saved model in a Streamlit application.
📊 Dataset
The project uses a Credit Card Fraud Detection dataset containing 284,807 transactions.
Dataset Information
Information
Value
Total Transactions
284,807
Total Features
30
Target Column
Class
Legitimate Transactions
284,315
Fraudulent Transactions
492
Class Distribution
Class
Meaning
Transactions
Percentage
0
Legitimate
284,315
99.83%
1
Fraudulent
492
0.17%
The dataset is therefore highly imbalanced.
Dataset Features
The dataset contains:
Time
V1 to V28
Amount
Class
The Class column is the target variable.
🤖 Machine Learning Model
The project uses Logistic Regression for binary classification.
Model Pipeline
Transaction Data
       ↓
Data Cleaning
       ↓
Train-Test Split
       ↓
StandardScaler
       ↓
Logistic Regression
       ↓
Prediction
       ↓
Fraud / Legitimate
Model Configuration
LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)
The class_weight="balanced" parameter helps the model give greater consideration to the minority fraud class.
📈 Model Performance
The model was evaluated on 56,962 test transactions.
Metric
Score
Accuracy
97.55%
Precision
6.10%
Recall
91.84%
F1-Score
11.44%
Classification Report
precision    recall  f1-score   support

           0     0.9999    0.9756    0.9876     56864
           1     0.0610    0.9184    0.1144        98

    accuracy                         0.9755     56962
   macro avg     0.5304    0.9470    0.5510     56962
weighted avg     0.9982    0.9755    0.9861     56962
Confusion Matrix
[[55478  1386]
 [    8    90]]
For the fraud class, the model correctly identified 90 out of 98 fraudulent transactions in the test set, giving a 91.84% recall.
Since this is a highly imbalanced classification problem, accuracy alone is not sufficient to evaluate the model. Precision, Recall, and F1-Score are also considered.
🛠️ Technologies Used
Programming Language
Python
Libraries
Pandas
NumPy
Scikit-learn
Joblib
Streamlit
Matplotlib
Seaborn
Machine Learning
Logistic Regression
StandardScaler
Train-Test Split
Classification Metrics
Confusion Matrix
Imbalanced Classification
Development Tools
VS Code
Git
GitHub
📂 Project Structure
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
Note: The original dataset is not included in this GitHub repository because the dataset file is larger than GitHub's 100 MB individual file limit. Download the dataset separately and place it inside the data folder.
⚙️ Installation
1. Clone the Repository
git clone https://github.com/jayaM01/Credit-Card-Fraud-Detection.git
2. Open the Project Folder
cd Credit-Card-Fraud-Detection
3. Install Required Libraries
pip install -r requirements.txt
📥 Dataset Setup
Create a data folder inside the project:
Credit-Card-Fraud-Detection/
└── data/
Place the dataset inside the folder:
data/creditcard.csv
The final dataset path should be:
Credit-Card-Fraud-Detection/data/creditcard.csv
🧠 Train the Model
To train the Machine Learning model, run:
python train_model.py
The training script performs the following steps:
Loads the dataset
Checks missing values
Displays class distribution
Separates features and target
Splits data into training and testing sets
Applies feature scaling
Trains Logistic Regression
Generates predictions
Calculates evaluation metrics
Displays the confusion matrix
Saves the trained model
The trained model is saved as:
Fraud Detection Model.pkl
🌐 Run the Streamlit Application
Start the application using:
streamlit run app.py
The Streamlit application provides an interactive interface for making transaction predictions.
Application Features
Transaction prediction
Fraud probability
Legitimate transaction probability
Interactive user interface
Example Prediction
Prediction: Legitimate Transaction

Legitimate Probability: 95.44%
Fraud Probability: 4.56%
The probabilities shown by the application are for an individual transaction and should not be interpreted as the overall model accuracy.
🔐 Disclaimer
This project is developed for educational and demonstration purposes.
It is not intended to replace production-grade financial fraud detection systems, banking security infrastructure, or professional financial decision-making systems.
🚀 Future Improvements
The project can be further improved by:
Implementing Random Forest
Implementing XGBoost or other advanced models
Improving precision while maintaining high fraud recall
Hyperparameter tuning
Cross-validation
Advanced techniques for handling class imbalance
ROC-AUC analysis
Precision-Recall curve analysis
Feature importance analysis
Real-time transaction monitoring
Database integration
Cloud deployment
Advanced Streamlit dashboard
Model monitoring and automatic retraining
🎓 Learning Outcomes
Through this project, I learned:
Data preprocessing
Data cleaning
Exploratory data analysis
Handling imbalanced datasets
Feature scaling
Binary classification
Logistic Regression
Model evaluation
Confusion Matrix interpretation
Precision, Recall, and F1-Score
Model serialization using Joblib
Streamlit application development
Git and GitHub project management
⭐ Project Highlights
Machine Learning based fraud detection
Handles highly imbalanced transaction data
Uses class-balanced Logistic Regression
Achieved 91.84% fraud recall
Evaluated using multiple classification metrics
Includes a Streamlit prediction application
Complete ML workflow from preprocessing to deployment
👩‍💻 Author
Jaya
B.Tech – Computer Science and Engineering
GitHub
https://github.com/JayaMishra01
LinkedIn
https://linkedin.com/in/jaya-1569033a
