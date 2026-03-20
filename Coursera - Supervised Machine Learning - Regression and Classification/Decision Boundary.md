# Logistic Regression — Decision Boundary

## Goal
- Visualize the decision boundary of a logistic regression model
- Understand how model parameters split the input space into class regions

## Logistic regression model
$$
f_{w,b}(x) = g(w \cdot x + b)
$$

where:
$$
g(z) = \frac{1}{1 + e^{-z}}
$$

## Prediction rule
- Predict class `1` if `f_{w,b}(x) >= 0.5`
- Predict class `0` if `f_{w,b}(x) < 0.5`

Equivalent boundary condition:
$$
w \cdot x + b = 0
$$

## Decision boundary
- The set of points where the model is exactly undecided
- Separates the feature space into class 0 and class 1 regions

## For two features
If:
$$
w = [w_1, w_2], \quad x = [x_1, x_2]
$$

then the boundary is:
$$
w_1 x_1 + w_2 x_2 + b = 0
$$

Solving for `x_2`:
$$
x_2 = -\frac{w_1}{w_2}x_1 - \frac{b}{w_2}
$$

## Plotting idea
1. Plot training examples with different markers for each class
2. Draw the line where `w·x + b = 0`
3. Check whether the line separates the classes well

## Core interpretation
- One side of the boundary → predicted class 0
- Other side of the boundary → predicted class 1
- If the boundary is linear, the classifier is linear in the input features

## Key takeaway
- Logistic regression predicts probabilities
- The decision boundary is where that probability is 0.5
- For two input features, the boundary is a straight line

---
## Tags

#classification #logistic_regression 