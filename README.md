
Credit Card Fraud Detection
📌 Project Overview
Credit Card Fraud Detection is a Machine Learning-based application designed to identify potentially fraudulent credit card transactions.
The project uses transaction data, preprocessing techniques, and a trained classification model to predict whether a transaction is legitimate or fraudulent.
🎯 Problem Statement
Credit card fraud can cause financial losses for customers and financial institutions. Detecting suspicious transactions automatically can help reduce fraudulent activities and improve transaction security.
💡 Solution
This project uses Machine Learning to analyze transaction features and classify transactions as:
Legitimate Transaction
Fraudulent Transaction
The trained model is integrated with a Streamlit application for easy prediction.
🛠️ Technologies Used
Python
Pandas
NumPy
Scikit-learn
Streamlit
Matplotlib
Seaborn
Pickle
🤖 Machine Learning
The project includes:
Data preprocessing
Exploratory Data Analysis
Feature analysis
Train-test split
Machine Learning model training
Model evaluation
Fraud prediction
The trained model is saved as:
Fraud Detection Model.pkl
📂 Project Structure
Credit-Card-Fraud-Detection/
│
├── app.py
├── train_model.py
├── Fraud Detection Model.pkl
├── requirements.txt
├── README.md
└── .gitignore
⚙️ How to Run
1. Clone the repository
git clone https://github.com/jayaMishra01/Credit-Card-Fraud-Detection.git
2. Open the project folder
cd Credit-Card-Fraud-Detection
3. Install dependencies
pip install -r requirements.txt
4. Run the Streamlit application
streamlit run app.py
The application will open in your browser.
📊 Dataset
The original dataset is not stored directly in this GitHub repository because it exceeds GitHub's 100 MB file-size limit.
The dataset is used locally for model training and is excluded through .gitignore.
🔮 Future Improvements
Improve fraud detection performance
Add additional Machine Learning models
Handle class imbalance using appropriate techniques
Add real-time transaction monitoring
Deploy the application online
Add model performance dashboards
👩‍💻 Author
Jaya 
Computer Science & Engineering Student
GitHub: https://github.com/jayaMishra01
