# Random Forest, XGBoost, and When to Use Trees

## Random Forest

### Bagged decision trees
A bagged decision tree ensemble is built by:

1. sampling with replacement to create a new training set
2. training one tree on that sample
3. repeating many times
4. combining the trees by voting

Typical number of trees:
- around 64
- 100
- 128

More trees usually help up to a point, then returns diminish.

---

### Random forest improvement
Random forest adds one more source of randomness.

At each node:
- instead of considering all features
- choose a random subset of features
- then pick the best split from that subset

This makes the trees more different from each other and improves robustness.

Typical choice:
- use about `sqrt(n)` features at each split when there are many features

---

## Why random forests work well
Random forest is more robust because:
- each tree sees a slightly different bootstrap sample
- each split sees a random subset of features
- the ensemble averages over many variations

---

## XGBoost

XGBoost is a very strong and widely used tree-ensemble method.

### Idea
Instead of giving equal attention to all training examples every time, XGBoost focuses more on examples that previous trees handled poorly.

This is similar to deliberate practice:
- identify mistakes
- focus next effort on those mistakes

### Why it is effective
- often faster and stronger than plain bagging
- has good default splitting and stopping behavior
- includes regularization to reduce overfitting

### Practical use
In code, you typically use:
- `XGBClassifier` for classification
- `XGBRegressor` for regression

---

## When to use tree ensembles vs neural networks

### Tree ensembles
Good for:
- tabular / structured data
- spreadsheet-like datasets
- classification and regression on structured features

Advantages:
- fast to train
- often strong on structured data
- small trees can be interpretable

### Neural networks
Good for:
- images
- audio
- video
- text
- mixed structured + unstructured data

Advantages:
- work well on unstructured data
- support transfer learning
- often better for large, complex inputs

---

## Practical rule
- Use tree ensembles for structured/tabular data
- Use neural networks for unstructured data
- If you need a strong default tree-ensemble method, use XGBoost

---

## Key takeaways
- Bagging builds many trees from bootstrap samples.
- Random forests add random feature selection at each split.
- XGBoost is a boosted tree method that focuses on previous mistakes.
- Tree ensembles are usually best for tabular data.
- Neural networks are usually best for unstructured data.

---
## Tags

#trees #deep-learning 