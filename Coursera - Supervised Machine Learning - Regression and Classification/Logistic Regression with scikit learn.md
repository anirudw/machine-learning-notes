# Logistic Regression with scikit-learn

## Goal
- Train a logistic regression model using scikit-learn
- Make predictions on the training set
- Check training accuracy

## Dataset
```python
import numpy as np

X = np.array([[0.5, 1.5],
              [1.0, 1.0],
              [1.5, 0.5],
              [3.0, 0.5],
              [2.0, 2.0],
              [1.0, 2.5]])

y = np.array([0, 0, 0, 1, 1, 1])
```

## Model
```python
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression()
lr_model.fit(X, y)
```

## Prediction
```python
y_pred = lr_model.predict(X)
```

For this dataset:
```python
[0 0 0 1 1 1]
```

## Accuracy
```python
lr_model.score(X, y)
```

For this dataset:
```python
1.0
```

## Key idea
- `fit(X, y)` trains the model
- `predict(X)` returns class labels
- `score(X, y)` returns accuracy

---
## Tags

#classification  #logistic_regression #scikit_learn