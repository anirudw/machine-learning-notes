## Goal
- Fit a [[Linear Regression]] model using `scikit-learn`
- Avoid manual cost/gradient descent implementation
- Predict outputs and inspect learned parameters

## Model
For a feature vector `x`:

$$
f(x) = w \cdot x + b
$$

For multiple examples:

$$
\mathbf{f} = Xw + b
$$

## Main API
```python
from sklearn.linear_model import LinearRegression
```

## Workflow
1. Prepare `X_train` and `y_train`
2. Create model
3. Fit model
4. Predict
5. Read coefficients and intercept

## Train
```python
model = LinearRegression()
model.fit(X_train, y_train)
```

## Predict
```python
y_pred = model.predict(X_train)
```

## Parameters
```python
w = model.coef_      # weights
b = model.intercept_ # bias
```

## Shapes
- `X_train`: `(m, n)`
- `y_train`: `(m,)`
- `model.coef_`: `(n,)`
- `model.intercept_`: scalar

## Interpretation
- `coef_` stores the learned weights
- `intercept_` stores the bias term
- The model is still linear in the input features

## Key idea
- Earlier labs: implement prediction, cost, and gradient descent manually
- This lab: let `scikit-learn` handle training internally

## Minimal example
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_train)
w = model.coef_
b = model.intercept_
```

## Takeaway
- Use `LinearRegression` when you want a fast, standard implementation
- The underlying model remains the same linear form
- The library handles optimization for you

---
## Tags

#regression #linear_regression  #scikit_learn 