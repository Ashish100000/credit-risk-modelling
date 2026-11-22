# Credit Risk Modelling & Credit Scorecard Development

An end-to-end machine learning project focused on predicting loan defaults and estimating borrower credit risk using historical Lending Club loan data.

## Project Overview

Credit risk assessment is an important part of lending operations. Financial institutions need to evaluate the likelihood of borrowers defaulting on their loans before making lending decisions.

In this project, I developed a **Logistic Regression classification model** to estimate the Probability of Default (PD) and translated the model's output into an illustrative credit score ranging from 300 to 850.

The project demonstrates key concepts used in credit risk analytics, including data preprocessing, exploratory data analysis, predictive modelling, model evaluation and credit scorecard development.

## Business Problem

How can historical borrower and loan information be used to identify borrowers who are more likely to default?

The objective is to build a model that estimates default probability and categorises borrowers into different risk levels, helping demonstrate how data analytics and machine learning can support credit risk assessment.

## Dataset

- **Source:** Lending Club historical loan data.
- **File:** `accepted_2007_to_2018Q4.csv.gz`
- **Target variable:** `loan_status_flag`
- **Target definition:**
  - `1` – Default / Charged Off
  - `0` – Fully Paid

The notebook reads the first 20,000 records and selects up to 10,000 resolved loan records for analysis and modelling.

**Dataset setup:** Download the Lending Club dataset separately and place `accepted_2007_to_2018Q4.csv.gz` in the same directory as the notebook. The dataset is not included in this repository because of its large file size.

## Tools & Technologies

- **Programming:** Python
- **Data Analysis:** Pandas, NumPy
- **Data Visualisation:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Model Persistence:** Joblib
- **Environment:** Jupyter Notebook

## Project Workflow

### 1. Data Loading & Preparation
- Loaded historical Lending Club loan records.
- Selected loans with known final outcomes.
- Created a binary target variable for loan default prediction.
- Removed identifiers, unsuitable fields and potential data leakage columns.

### 2. Exploratory Data Analysis (EDA)
- Examined dataset structure and data types.
- Analysed missing values and duplicate records.
- Studied the distribution of default and non-default loans.
- Explored numerical features and default rates across selected categories.

### 3. Data Preprocessing
- Separated training and testing datasets.
- Handled missing numerical values using median imputation.
- Handled missing categorical values using most-frequent imputation.
- Applied standard scaling to numerical features.
- Applied one-hot encoding to categorical features.

Preprocessing transformations were fitted within a Scikit-learn pipeline using training data to help prevent data leakage.

### 4. Model Development
- Developed a Logistic Regression classification model.
- Trained the model using historical loan records.
- Generated default predictions and Probability of Default (PD) estimates on unseen test data.

### 5. Model Evaluation

Evaluated model performance using the following metrics:

- **ROC-AUC:** Measures the model's ability to distinguish between default and non-default borrowers.
- **Gini Coefficient:** Measures discriminatory power and is calculated as `2 × AUC − 1`.
- **KS Statistic:** Measures the maximum separation between cumulative default and non-default distributions.
- **Brier Score:** Evaluates the accuracy of predicted probabilities.

Also generated a classification report, confusion matrix and ROC curve.

### 6. Credit Scorecard Development

Converted model output into an illustrative credit score ranging from **300 to 850**.

Scorecard assumptions:
- Base Score: 650
- Points to Double the Odds (PDO): 40
- Higher scores indicate lower predicted default risk.

### 7. Risk Segmentation

Grouped borrowers into four illustrative risk categories:

| Credit Score | Risk Category |
|---|---|
| 760+ | Low Risk |
| 710–759 | Medium Risk |
| 660–709 | High Risk |
| Below 660 | Very High Risk |

Analysed the observed default rate and average predicted PD across these risk categories.

### 8. Model Deployment Demonstration
- Saved the complete preprocessing and Logistic Regression pipeline using Joblib.
- Demonstrated scoring an individual applicant.
- Generated the applicant's predicted Probability of Default, illustrative credit score and risk category.

## Repository Structure

```text
credit-risk-modelling/
│
├── credit_risk_modelling.ipynb
├── README.md
├── requirements.txt
├── .gitignore
│
└── accepted_2007_to_2018Q4.csv.gz  (download separately)
```

## How to Run the Project

**1. Clone the repository**

```bash
git clone https://github.com/Ashish100000/credit-risk-modelling-Copy.git
```

**2. Navigate to the project directory**

```bash
cd credit-risk-modelling-Copy
```

**3. Install the required libraries**

```bash
pip install -r requirements.txt
```

**4. Download the dataset**

Obtain the Lending Club historical loan dataset and place `accepted_2007_to_2018Q4.csv.gz` in the project directory.

**5. Open the Jupyter Notebook**

```bash
jupyter notebook
```

Open `credit_risk_modelling.ipynb` and execute the cells sequentially.

## Limitations & Future Improvements

- Use the complete dataset and a time-based validation strategy.
- Perform probability calibration and calibration analysis.
- Review predictors for availability at the actual loan decision timestamp.
- Evaluate model fairness, stability and potential data drift.
- Optimise risk thresholds based on lending business requirements.
- Validate the scorecard on an independent out-of-time dataset.

## Disclaimer

This is an educational machine learning and portfolio project. The credit score scale and risk categories are illustrative and are not equivalent to an officially validated credit bureau score or a production-ready lending decision system.

---

**Author:** Ashish Upadhyay

**Focus Areas:** Credit Risk Analytics | Data Analytics | Machine Learning | FinTech
