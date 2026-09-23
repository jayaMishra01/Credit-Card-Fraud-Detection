
Credit Card Fraud Detection
A Machine Learning project that detects potentially fraudulent credit card transactions using Python, Scikit-learn and Logistic Regression.
The project handles the highly imbalanced nature of fraud detection using class balancing and evaluates the model using Accuracy, Precision, Recall and F1-Score.
📌 Project Overview
Credit card fraud is a major problem in digital payments. Fraudulent transactions are rare compared to legitimate transactions, making fraud detection a challenging imbalanced classification problem.
This project builds a machine learning model that analyzes transaction features and predicts whether a transaction is:
0 → Legitimate Transaction
1 → Fraudulent Transaction
The trained model is also integrated with a Streamlit web application for interactive predictions.
🎯 Problem Statement
The objective of this project is to develop a machine learning system that can identify fraudulent credit card transactions while minimizing missed fraudulent transactions.
Since fraudulent transactions represent only a very small percentage of the dataset, the model must handle class imbalance effectively.
💡 Solution
The project follows a complete Machine Learning pipeline:
Load the transaction dataset.
Check dataset structure and missing values.
Analyze class distribution.
Separate features and target variable.
Split the data into training and testing sets.
Apply feature scaling using StandardScaler.
Train a Logistic Regression classification model.
Handle class imbalance using class_weight="balanced".
Generate predictions on the test dataset.
Evaluate the model using multiple classification metrics.
Save the trained model using Joblib.
Use the saved model in a Streamlit application.
📊 Dataset
The project uses the Credit Card Fraud Detection dataset.
Dataset Information
Total transactions: 284,807
Features: 30
Target column: Class
Legitimate transactions: 284,315
Fraudulent transactions: 492
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
Fraud
492
0.17%
This shows that the dataset is highly imbalanced.
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
Input Transaction
       ↓
Data Cleaning
       ↓
Train-Test Split
       ↓
StandardScaler
       ↓
Logistic Regression
       ↓
Class Prediction
       ↓
Fraud / Legitimate
Model Configuration
LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)
class_weight="balanced" is used because fraudulent transactions are much less frequent than legitimate transactions.
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
The model detected 90 out of 98 fraudulent transactions in the test set, resulting in a 91.84% fraud recall.
Because fraud detection is a highly imbalanced classification problem, Accuracy alone is not sufficient for evaluating the model. Precision, Recall and F1-Score are also considered.
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
Note: The original dataset is not included in this GitHub repository because the dataset file exceeds GitHub's 100 MB individual file limit. Download the dataset separately and place it inside the data folder.
⚙️ Installation
1. Clone the Repository
git clone https://github.com/jayaM01/Credit-Card-Fraud-Detection.git
2. Open the Project
cd Credit-Card-Fraud-Detection
3. Install Required Libraries
pip install -r requirements.txt
📥 Dataset Setup
Create a data folder inside the project:
Credit-Card-Fraud-Detection/
└── data/
Place the downloaded dataset inside it:
data/creditcard.csv
The expected path is:
data/creditcard.csv
🧠 Train the Model
To train the model again, run:
python train_model.py
The script will:
Load the dataset
Check missing values
Display class distribution
Split the dataset
Scale the features
Train Logistic Regression
Evaluate the model
Display the confusion matrix
Save the trained model
The trained model will be saved as:
Fraud Detection Model.pkl
🌐 Run the Streamlit Application
Start the web application using:
streamlit run app.py
The Streamlit application allows users to enter transaction information and receive a prediction.
Application Output
The application provides:
Transaction prediction
Fraud probability
Legitimate transaction probability
Interactive user interface
Example:
Prediction: Legitimate Transaction

Legitimate Probability: 95.44%
Fraud Probability: 4.56%
These probabilities are for an individual transaction entered into the application and should not be interpreted as the overall model accuracy.
🔐 Important Note
This project is developed for educational and demonstration purposes.
It is not intended to replace production-grade financial fraud detection systems or real banking security infrastructure.
🚀 Future Improvements
The project can be improved by:
Using advanced models such as Random Forest, XGBoost or LightGBM
Improving precision while maintaining high fraud recall
Applying advanced techniques for imbalanced datasets
Hyperparameter tuning
Cross-validation
ROC-AUC and Precision-Recall curve analysis
Feature importance analysis
Real-time transaction monitoring
Cloud deployment
Database integration
Improved Streamlit dashboard
Model monitoring and retraining pipeline
🎓 Learning Outcomes
Through this project, I learned:
Data preprocessing
Exploratory data analysis
Handling imbalanced datasets
Feature scaling
Binary classification
Logistic Regression
Model evaluation
Confusion matrix interpretation
Precision, Recall and F1-Score
Model serialization using Joblib
Streamlit application development
Git and GitHub project management
👩‍💻 Author
Jaya
B.Tech – Computer Science and Engineering
GitHub
https://github.com/JayaMishra01
⭐ Project Highlights
Machine Learning based fraud detection
Handles highly imbalanced transaction data
Uses class-balanced Logistic Regression
Achieved 91.84% fraud recall
Includes model evaluation using multiple metrics
Includes a Streamlit prediction application
Complete ML pipeline from data preprocessing to deployment
