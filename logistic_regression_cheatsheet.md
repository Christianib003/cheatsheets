# Logistic Regression: Comprehensive Implementation Guide

This notebook provides a step-by-step guide for implementing logistic regression for binary classification using Python and `scikit-learn`. It mirrors the structure of a comprehensive linear regression guide, covering problem definition, data loading, exploration, preprocessing, feature engineering, normalization, model training, evaluation, interpretation, and visualization of the cost function and learning curve. Each section includes code, explanations of industry practices, and the underlying logic to ensure the guide is practical and educational.

---

## 1. Problem Definition and Setup

### Objective
Predict a binary target variable \( y \) (e.g., 0 or 1) using input features \( X \). Logistic regression models the probability \( p(y=1|X) \) using the logistic function.

### Why Logistic Regression?
Logistic regression is ideal for binary classification tasks. Unlike linear regression, which predicts continuous values, logistic regression outputs probabilities between 0 and 1, making it suitable for problems like spam detection, customer churn prediction, or disease diagnosis.

### Code
```python
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(42)

# Define the problem
print("Objective: Predict whether a customer will churn (1) or not (0) based on features like tenure, monthly charges, etc.")
```

### Underlying Logic
Logistic regression models the probability using the logistic function: \( p(y=1|X) = \sigma(X\beta) \), where \( \sigma(z) = \frac{1}{1 + e^{-z}} \). It maximizes the likelihood of the observed data by minimizing the log loss (cross-entropy).

---

## 2. Data Collection and Loading

### Objective
Load the dataset into a `pandas` DataFrame and verify its structure.

### Code
```python
# Load dataset (replace 'path_to_dataset.csv' with your file path)
data = pd.read_csv('path_to_dataset.csv')

# Display first few rows
print(data.head())
```

### Notes
- Replace `'path_to_dataset.csv'` with the path to your dataset.
- Ensure the dataset includes a binary target column (e.g., 0/1 for churn/no churn).

---

## 3. Data Exploration and Analysis

### Objective
Understand the dataset’s structure, distributions, and relationships to ensure it’s suitable for logistic regression.

### Code
```python
# Summary statistics
print(data.describe())

# Check for missing values
print(data.isnull().sum())

# Visualize target distribution
sns.countplot(x='target', data=data)
plt.title('Target Distribution')
plt.show()

# Visualize feature-target relationships (replace feature names)
sns.boxplot(x='target', y='feature1', data=data)
plt.title('Feature1 vs Target')
plt.show()

# Correlation matrix
corr = data.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
```

### Explanation
- **Summary Statistics**: Check feature ranges and target balance.
- **Missing Values**: Identify any preprocessing needs.
- **Visualizations**: 
  - A countplot shows class balance (e.g., balanced or imbalanced classes).
  - Boxplots reveal feature-target relationships.
  - A correlation matrix detects multicollinearity among features.

### Underlying Logic
Logistic regression assumes a linear relationship between features and the log-odds of the target. EDA helps verify this assumption and informs preprocessing steps.

---

## 4. Data Preprocessing

### Objective
Prepare the data by handling missing values and encoding categorical variables.

### Code
```python
# Handle missing values (e.g., impute with mean)
data['feature1'].fillna(data['feature1'].mean(), inplace=True)

# Encode categorical variables (replace 'categorical_feature')
data = pd.get_dummies(data, columns=['categorical_feature'], drop_first=True)

# Separate features and target
X = data.drop('target', axis=1)
y = data['target']
```

### Explanation
- **Imputation**: Use the mean for continuous features to handle missing data simply.
- **Encoding**: One-hot encoding converts categorical variables into numerical format, with `drop_first=True` to avoid multicollinearity.

### Industry Practice
Preprocessing ensures the data is clean and compatible with logistic regression, which requires numerical inputs.

---

## 5. Feature Engineering

### Objective
Enhance model performance by creating or transforming features.

### Code
```python
# Create interaction term
data['feature1_feature2'] = data['feature1'] * data['feature2']

# Create polynomial feature
data['feature1_squared'] = data['feature1'] ** 2

# Update X with new features
X = data.drop('target', axis=1)
```

### Explanation
- **Interaction Terms**: Capture combined effects of features (e.g., `feature1 * feature2`).
- **Polynomial Features**: Model non-linear relationships within the logistic framework.

### Underlying Logic
Although logistic regression assumes linearity in the log-odds, feature engineering can indirectly capture non-linear patterns, improving predictive power.

---

## 6. Normalization

### Objective
Scale features to have zero mean and unit variance for better model performance.

### Why Normalize?
Normalization ensures features contribute equally to the model and is critical for gradient-based methods (e.g., SGD) to converge efficiently.

### Code
```python
# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert back to DataFrame (optional, for readability)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
```

### Underlying Logic
In gradient descent, parameter updates \( \beta_j \gets \beta_j - \eta \cdot \frac{\partial J}{\partial \beta_j} \) depend on feature scales. Normalization stabilizes these updates.

---

## 7. Model Training

### Objective
Train the logistic regression model using `LogisticRegression` or `SGDClassifier` (for cost function visualization).

### Option 1: Using LogisticRegression
```python
# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train logistic regression
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
```

### Option 2: Using SGDClassifier (for Cost Function Plotting)
```python
# Initialize SGDClassifier with logistic loss
model_sgd = SGDClassifier(loss='log', max_iter=1, tol=None, warm_start=True, eta0=0.01, random_state=42)

# Train and track loss
n_iterations = 100
losses = []

for _ in range(n_iterations):
    model_sgd.partial_fit(X_train, y_train, classes=[0, 1])
    # Compute log loss (cross-entropy)
    prob = model_sgd.predict_proba(X_train)
    loss = -np.mean(y_train * np.log(prob[:, 1]) + (1 - y_train) * np.log(prob[:, 0]))
    losses.append(loss)

# Plot cost function
plt.plot(range(n_iterations), losses)
plt.xlabel('Iterations')
plt.ylabel('Training Loss (Log Loss)')
plt.title('Cost Function over Iterations')
plt.show()
```

### Explanation
- **`LogisticRegression`**: Uses an optimized solver (e.g., LBFGS) for efficient training.
- **`SGDClassifier`**: Enables incremental training and loss tracking with `loss='log'` for logistic regression.
- **Parameters**:
  - `eta0`: Learning rate.
  - `partial_fit`: Updates the model iteratively.

### Underlying Logic
Logistic regression minimizes the log loss: \( J(\beta) = -\frac{1}{m} \sum_{i=1}^m [y_i \log(p_i) + (1 - y_i) \log(1 - p_i)] \), where \( p_i = \sigma(X_i \beta) \).

---

## 8. Model Evaluation

### Objective
Assess the model’s performance using multiple metrics and visualize the ROC curve.

### Code
```python
# Predict on test set
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-score: {f1:.4f}')
print(f'ROC-AUC: {roc_auc:.4f}')

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

### Explanation
- **Metrics**:
  - **Accuracy**: Proportion of correct predictions.
  - **Precision**: True positives over predicted positives.
  - **Recall**: True positives over actual positives.
  - **F1-score**: Balances precision and recall.
  - **ROC-AUC**: Measures the model’s ability to distinguish classes.

### Industry Practice
Multiple metrics are used because accuracy alone can be misleading (e.g., with imbalanced data).

---

## 9. Learning Curve

### Objective
Visualize how model performance changes with training set size to diagnose bias and variance.

### Code
```python
# Generate learning curve
train_sizes, train_scores, val_scores = learning_curve(
    LogisticRegression(random_state=42),
    X_scaled, y, cv=5, scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 10)
)

# Calculate mean and standard deviation
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
val_scores_mean = np.mean(val_scores, axis=1)
val_scores_std = np.std(val_scores, axis=1)

# Plot learning curve
plt.figure()
plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training accuracy')
plt.plot(train_sizes, val_scores_mean, 'o-', color='g', label='Validation accuracy')
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color='r')
plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                 val_scores_mean + val_scores_std, alpha=0.1, color='g')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title('Learning Curve')
plt.legend(loc='best')
plt.show()
```

### Explanation
- **Training Accuracy (Red)**: Performance on training data.
- **Validation Accuracy (Green)**: Performance on cross-validation data.
- **Diagnosis**:
  - Large gap: Overfitting (high variance).
  - Low accuracy: Underfitting (high bias).

---

## 10. Model Interpretation

### Objective
Interpret the model’s coefficients to understand feature impacts.

### Code
```python
# Display coefficients
coefficients = pd.DataFrame(model.coef_[0], X.columns, columns=['Coefficient'])
print(coefficients)

# Calculate odds ratios
odds_ratios = np.exp(coefficients)
print(odds_ratios)
```

### Explanation
- **Coefficients**: A positive coefficient increases the log-odds of \( y=1 \); a negative one decreases it.
- **Odds Ratios**: \( e^{\beta_j} \) indicates the change in odds for a one-unit increase in the feature (e.g., 1.5 means a 50% increase in odds).

### Underlying Logic
The model predicts \( \log\left(\frac{p}{1-p}\right) = X\beta \), where coefficients \( \beta \) represent feature importance.

---

## 11. Model Deployment (Optional)

### Objective
Save the trained model and scaler for future use.

### Code
```python
import joblib
joblib.dump(model, 'logistic_regression_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
```

### Industry Practice
Saving both the model and scaler ensures consistent preprocessing in production.

---

## Conclusion

This notebook provides a complete logistic regression implementation, including:
- **Feature Engineering**: Interaction and polynomial features to capture complex patterns.
- **Normalization**: Using `StandardScaler` for stable convergence.
- **Cost Function Plot**: Via `SGDClassifier` to visualize training loss.
- **Learning Curve**: To assess model fit and generalization.

### How to Use
1. Replace `'path_to_dataset.csv'`, `'feature1'`, `'target'`, etc., with your dataset specifics.
2. Adjust `eta0` or `n_iterations` in `SGDClassifier` if the loss plot needs tuning.
3. Ensure the scaler is applied to new data when deploying the model.

This template is adaptable to various binary classification tasks—happy modeling!