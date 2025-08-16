# ❤️ Heart Disease Prediction using Machine Learning

## 📌 Project Description
This project predicts whether a patient is likely to have **heart disease** using the **UCI Heart Disease dataset**.  
The notebook contains complete steps from **data preprocessing, exploratory data analysis (EDA), feature engineering, and model building** to **performance evaluation**.  

The goal is to compare different **machine learning algorithms** and find the best-performing model for heart disease classification.

---

## 👨‍💻 Author
- **Name:** Kumavat Pravin Suresh  
- **Date:** 12 June 2025  
- **LinkedIn:** [Pravin Kumavat](https://www.linkedin.com/in/pravin-kumavat-76072531b/)  
- **GitHub:** [PravinKumavat252](https://github.com/PravinKumavat252)  
- **Kaggle:** [Pravin Kumavat](https://www.kaggle.com/pravinkumawat)  

---

## 📂 Dataset Information
- **Source:** UCI Heart Disease Dataset  
- **Instances:** ~303 patients  
- **Features:** 14 medical attributes + 1 target  
- **Target Variable:** `num` → indicates presence (1) or absence (0) of heart disease.  

### Attributes:
- `age` → Age of patient  
- `sex` → Gender (1 = Male, 0 = Female)  
- `cp` → Chest pain type (4 types)  
- `trestbps` → Resting blood pressure  
- `chol` → Serum cholesterol level  
- `fbs` → Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)  
- `restecg` → Resting electrocardiographic results  
- `thalach` → Maximum heart rate achieved  
- `exang` → Exercise-induced angina (1 = yes, 0 = no)  
- `oldpeak` → ST depression induced by exercise  
- `slope` → Slope of the peak exercise ST segment  
- `ca` → Number of major vessels (0–3)  
- `thal` → Thalassemia (normal/fixed defect/reversible defect)  
- `num` → Target (0 = No heart disease, 1 = Heart disease)  

---

## 🎯 Project Objectives
1. Perform **Exploratory Data Analysis (EDA)** to understand patterns and correlations.  
2. Handle missing values, outliers, and perform feature engineering.  
3. Apply **data preprocessing** (scaling, encoding, imputation).  
4. Train and evaluate multiple **machine learning models**.  
5. Compare model performances using accuracy, precision, recall, F1-score, and confusion matrices.  
6. Identify the **best classifier** for predicting heart disease.  

---

## 🛠️ Technologies Used
- **Language:** Python  
- **Libraries:**  
  - Data Handling: `pandas`, `numpy`  
  - Visualization: `matplotlib`, `seaborn`, `plotly`  
  - ML Models: `sklearn`, `xgboost`  
  - Preprocessing: `LabelEncoder`, `StandardScaler`, `Imputer`  

---

## ⚙️ Project Workflow
1. **Data Collection** – Load UCI Heart Disease dataset.  
2. **Data Preprocessing** – Handle missing values, scale features, encode categorical variables.  
3. **EDA & Visualization** – Distribution plots, correlation heatmap, feature relationships.  
4. **Model Training** – Logistic Regression, Decision Tree, Random Forest, SVM, KNN, Naive Bayes, Gradient Boosting, AdaBoost, XGBoost.  
5. **Model Evaluation** – Compare results using confusion matrix & classification report.  
6. **Model Selection** – Select best-performing classifier.  

---

## 📊 Results
- Achieved **~87% accuracy** on test data.  
- **Random Forest, Gradient Boosting, and XGBoost** gave the best results with balanced precision and recall.  
- Logistic Regression and KNN also performed decently.  

---

## 🚀 Future Enhancements
- Perform **hyperparameter tuning** for optimization.  
- Try **Deep Learning (ANN)** models.  
- Deploy the model using **Flask/Streamlit** as a web application.  
- Integrate with **real-time patient data** for practical usage.  

---

## 📜 References
- UCI Machine Learning Repository: Heart Disease Dataset  
- Research Paper: Detrano et al. (1989) – *Probability algorithm for diagnosis of coronary artery disease*  
- Kaggle community notebooks  

---

## 🏷️ License
This project is open-source under the **MIT License**.

