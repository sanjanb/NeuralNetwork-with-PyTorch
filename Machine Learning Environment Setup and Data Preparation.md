# Machine Learning Environment Setup and Data Preparation

## Lesson Objectives

- Install and set up essential libraries for machine learning.
- Load and explore an insurance charges dataset.
- Visualize data to understand feature relationships.
- Split the dataset into training and test sets.

## Lesson Plan

### 1. Library Setup

**Objective:** Install and set up essential libraries within a virtual environment.

**Steps:**

1. **Create a Virtual Environment:**
   - Ensure you have Python installed.
   - Create a virtual environment to manage dependencies.

   ```bash
   python -m venv ml_env
   ```

2. **Activate the Virtual Environment:**
   - On Windows:

     ```bash
     ml_env\Scripts\activate
     ```

   - On macOS/Linux:

     ```bash
     source ml_env/bin/activate
     ```

3. **Install Necessary Libraries:**
   - Use `pip` to install the required libraries.

   ```bash
   pip install scikit-learn torch torchvision torchmetrics pandas seaborn matplotlib
   ```

### 2. Dataset Introduction

**Objective:** Load and explore the insurance charges dataset.

**Steps:**

1. **Load the Dataset:**
   - Use `pandas` to load the dataset from a CSV file.

   ```python
   import pandas as pd

   df = pd.read_csv('insurance_charges.csv')
   ```

2. **Explore the Data Structure:**
   - Inspect the first few rows and data types.

   ```python
   print(df.head())
   print(df.info())
   ```

### 3. Data Visualization

**Objective:** Use visualization techniques to understand the data distribution and relationships.

**Steps:**

1. **Histograms and Boxplots:**
   - Visualize the distribution of insurance charges.
   - Analyze the relationship between features (age, smoking status) and charges.

   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt

   sns.histplot(df['charges'], kde=True)
   plt.title('Distribution of Insurance Charges')
   plt.show()

   sns.boxplot(x='smoker', y='charges', data=df)
   plt.title('Insurance Charges by Smoking Status')
   plt.show()
   ```

### 4. Data Splitting

**Objective:** Split the dataset into training and test sets.

**Steps:**

1. **Split the Data:**
   - Use `train_test_split` from `scikit-learn` to divide the dataset.

   ```python
   from sklearn.model_selection import train_test_split

   X = df.drop('charges', axis=1)
   y = df['charges']

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   ```

## Notes

- **Virtual Environment:** Isolates project dependencies to avoid conflicts.
- **Data Exploration:** Essential for understanding feature relationships and data distribution.
- **Visualization:** Helps identify key features influencing insurance charges.
- **Data Splitting:** Prepares data for model training and evaluation.
