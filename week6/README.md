# 🏦 Bank Institution Term Deposit Predictive Model
This project focuses on building a machine learning model to predict whether a customer will subscribe to a term deposit product based on a real-world bank marketing dataset. The solution demonstrates a full data science workflow from data exploration to model prediction, with a focus on handling imbalanced data and selecting the best-performing model.

## 📌 Project Objective
To predict customer subscription to bank term deposits using supervised machine learning techniques, with the goal of supporting marketing decision-making and campaign targeting.

## 🗃️ Dataset
* Source: [UCI Machine Learning Repository - Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)
* File Used: `bank-additional-full.csv`
* Target Variable: `y` – Whether the customer subscribed to a term deposit (`yes`/`no`)

## 🛠 Tools & Libraries
* Languages: Python
* Models: XGBoost, Logistic Regression, Multi-Layer Perceptron (MLP)
* Libraries: Pandas, NumPy, Scikit-learn, XGBoost, imbalanced-learn, Matplotlib, Seaborn
* Visualization: Python (Matplotlib, Seaborn), Tableau (Univariate and Bivariate Analysis)
* Others: SMOTE, PCA, t-SNE

## 📌 Project Workflow Breakdown

### 🔍 1. Exploratory Data Analysis (EDA)
EDA was conducted to understand data distribution, detect outliers, and uncover insights:
* Shape, structure, and summary statistics
* Categorical vs. numerical feature analysis
* Visualization with Tableau for univariate and bivariate insights
* Correlation between data features using heatmap
* Outlier treatment using the IQR method
* Identified target variable imbalance (y: yes/no).

### 🧼 2. Data Preprocessing
* Categorical variables encoding using One-Hot encoding
* Rescaling numerical variables using `StandardScaler()` to normalize features.
* Specifying dependent and independent variables
* Splitting the dataset into training and testing sets to properly evaluate model performance.
* Feature Engineering: Combined and refined features for better performance.
* Dimensionality Reduction: Applied Principal Component Analysis (PCA) to reduce feature space while retaining variance.
* Handling class imbalance: Applied SMOTE (Synthetic Minority Oversampling Technique) to balance the dataset.

### 🤖 3. Model Building
* Trained the following machine learning models:
  * Logistic Regression
  * XGBoost Classifier
  * Multi-Layer Perceptron (MLP)
* Implemented two types of cross-validation:
  * K-Fold Cross-Validation
  * Stratified K-Fold Cross-Validation
 
### 📈 4. Evaluation Metrics
* Assessed model performance using:
  * Accuracy
  * Area Under the Curve (AUC)
  * Precision
  * Recall
  * F1 Score
 
### 🎯 5. Model Comparison & Selection
* Compared results across all three models.
* XGBoost showed the best performance and was chosen as the final model for prediction.

| K-Fold Cross Validation Results |
|:-------------------------------:|

| Model                   | Accuracy (%) | AUC (%)  | Precision (%) | Recall (%) | F1-Score (%) |
|-------------------------|--------------|----------|---------------|------------|--------------|
| Logistic Regression     | 70.74        |     -    | 61.98         | 63.84      | 57.32        |
| XGBoost                 | 85.53        |     -    | 69.48         | 78.39      | 71.70        |
| Multi Layer Perceptron  | 89.57        | 76.85    | 56.86         | 26.81      | 36.20        |

| Stratified K-Fold Cross Validation Results |
|:------------------------------------------:|

| Model                   | Accuracy (%) | AUC (%)  | Precision (%) | Recall (%) | F1-Score (%) |
|-------------------------|--------------|----------|---------------|------------|--------------|
| Logistic Regression     | 73.46        | 79.01    | 78.71         | 64.31      | 70.78        |
| XGBoost                 | **87.23**    | **94.55**| **88.60**     | **85.43**  | **86.88**    |
| Multi Layer Perceptron  | 89.53        | 77.05    | 56.58         | 27.47      | 36.82        |

### 🔮 6. Final Prediction
* Used the trained XGBoost model to make predictions on unseen data.
* Output includes binary classification (`yes`/`no`) for term deposit subscription likelihood.

## 📁 Project Structure
```
.
├── notebooks/
│   ├── WEEK6_CHALLENGE_EDA.ipynb   # Exploratory Data Analysis
│   └── WEEK6_CHALLENGE_MODEL.ipynb # Script conversion: preprocessing to prediction
├── Data.py                         # Define dependent/independent variables, split data, apply SMOTE & PCA
├── Main.py                         # Main script to run pipeline on dataset without outliers
├── Main-org.py                     # Main script to run pipeline on original dataset
├── Model.py                        # Build models (LogReg, XGBoost, MLP), cross-validation, evaluation
├── Preprocessing.py                # Encoding categorical columns, rescaling numerical columns
├── README.md                       # Project documentation
├── Util.py                         # python library to ignore warnings
└── requirements.txt                # Project dependencies
```

## 📌 How to Run
1. Install dependencies
```pip install -r requirements.txt```
2. Run the main script
```python Main.py```

## 📬 Contact
- For questions, feedback, opportunities, or collaborations, connect with me via [LinkedIn](https://www.linkedin.com/in/glory-odeyemi/).
- For more exciting projects or inspiration, check out my [Portfolio](https://gloryodeyemi.github.io/) or [GitHub Repositories](https://github.com/gloryodeyemi).
