# Support Vector Machine (SVM) for Breast Cancer Prediction

## Introduction
Support Vector Machine (SVM) is a type of supervised learning algorithm used for classification and regression. In this project, we use SVM to predict whether a breast cancer tumor is malignant (bad type) or benign (good type) using the breast cancer dataset from the Scikit-learn library.

## Dataset
We use the breast cancer dataset available in Scikit-learn.

## Steps

### 1. Import Required Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```

### 2. Load Dataset
```python
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
```

### 3. Explore the Dataset
```python
print(cancer.keys())  # View dataset keys
print(cancer['DESCR'])  # Description of the dataset
```

### 4. Create DataFrame
Create a DataFrame for features and target.
```python
df_feat = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
df_target = pd.DataFrame(cancer['target'], columns=['cancer'])
df = pd.concat([df_feat, df_target], axis=1)
```

### 5. Data Exploration
Visualize the data to understand the relationships between features and the target.
```python
sns.scatterplot(x='mean concavity', y='mean texture', hue='cancer', data=df)
sns.scatterplot(x='worst texture', y='mean texture', hue='cancer', data=df)
sns.scatterplot(x='mean radius', y='mean perimeter', hue='cancer', data=df)
sns.countplot(x='cancer', data=df)
```

### 6. Split the Data
Split the data into training and testing sets.
```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df_feat, np.ravel(df_target), test_size=0.3, random_state=101)
```

### 7. Train the SVM Model
```python
from sklearn.svm import SVC
model = SVC()
model.fit(x_train, y_train)
```

### 8. Evaluate the Model
Make predictions and evaluate the model using confusion matrix and classification report.
```python
from sklearn.metrics import classification_report, confusion_matrix
predictions = model.predict(x_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
```

### 9. Hyperparameter Tuning with Grid Search
Use Grid Search to find the best parameters for the model.
```python
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
grid.fit(x_train, y_train)
grid_predictions = grid.predict(x_test)
print(confusion_matrix(y_test, grid_predictions))
print(classification_report(y_test, grid_predictions))
print(grid.best_params_)
```

## Conclusion
- We used the SVM method to classify breast cancer tumors.
- We utilized Grid Search to optimize model parameters.
- The model's accuracy improved from 0.92 to 0.94 after hyperparameter tuning.

## Summary
This project demonstrates the use of SVM for classification tasks, particularly in predicting breast cancer. We leveraged the Grid Search technique to fine-tune our model's hyperparameters, resulting in better performance. This approach can be applied to other machine learning models with hyperparameters.
