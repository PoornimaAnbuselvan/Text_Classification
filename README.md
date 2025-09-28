# Text_Classification
This project classifies consumer complaints into four categories: Credit reporting, Debt collection, Consumer Loan, and Mortgage. Using text preprocessing (cleaning, tokenization, TF-IDF) and models like Logistic Regression, Naïve Bayes, and SVM, the system predicts complaint types with high accuracy.
# Consumer Complaint Text Classification

## 📖 Overview
This project performs **multi-class text classification** on the [Consumer Complaint Database](https://catalog.data.gov/dataset/consumer-complaint-database).  
It categorizes consumer complaints into four classes:
- 0: Credit reporting, repair, or other
- 1: Debt collection
- 2: Consumer Loan
- 3: Mortgage

The aim is to help financial institutions automatically identify the nature of complaints for faster resolution.

---

## ⚙️ Steps in the Project
1. **Exploratory Data Analysis (EDA) & Feature Engineering**  
   - Analyzed dataset, checked distribution, and mapped categories.  

2. **Text Pre-Processing**  
   - Lowercasing, punctuation & stop-word removal.  
   - Tokenization and lemmatization.  
   - Text → numeric vectors using TF-IDF.  

3. **Model Selection**  
   - Compared Logistic Regression, Naïve Bayes, Random Forest, and SVM.  

4. **Model Performance Comparison**  
   - Evaluated models on Accuracy, Precision, Recall, F1-score.  

5. **Model Evaluation**  
   - Chose the best-performing model based on metrics and confusion matrix.  

6. **Prediction**  
   - Final model predicts complaint category for unseen text.  

---
