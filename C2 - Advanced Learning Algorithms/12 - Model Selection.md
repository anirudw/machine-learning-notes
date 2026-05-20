
## Goal
Choose a model that generalizes well without using the test set during model selection.

---

## 1. Why a separate evaluation set is needed

Training error is usually too optimistic because the model is fit to the training set.

So, to compare models, use a separate **cross-validation set**.

The **test set** is reserved for the final, unbiased estimate of generalization performance.

---

## 2. Data split

Split the dataset into three parts:

- **Training set**: fit model parameters
- **Cross-validation set**: compare models / tune hyperparameters
- **Test set**: report final performance once

Common labels:

- `train`
- `cv` or `dev`
- `test`

---

## 3. Error definitions

### Regression
Use mean squared error:

$$
J = \frac{1}{m}\sum_{i=1}^{m} \left(f(x^{(i)}) - y^{(i)}\right)^2
$$

### Classification
Use misclassification error:

$$
\text{error} =
\frac{\text{number of misclassified examples}}{m}
$$

### Important
- Do **not** include the regularization term in these evaluation errors.
---

## 4. Model selection procedure

1. Train multiple candidate models on the training set.
2. Compute cross-validation error for each model.
3. Choose the model with the lowest cross-validation error.
4. Evaluate the chosen model once on the test set.

Do **not** use the test set to decide which model is best.

---

## 5. Example: polynomial regression

Try several degrees:

- `d = 1`
- `d = 2`
- `d = 3`
- ...
- `d = 10`

For each degree:
- fit on the training set
- compute training error
- compute cross-validation error

Choose the degree with the lowest `J_cv`.

Then report the final `J_test` only once.

---

## 6. Example: neural network selection

You can also compare:

- number of layers
- number of units per layer
- other architecture choices

Train each candidate on the training set, compare on the cross-validation set, and keep the best one.

---

## 7. Feature scaling
## Feature Scaling

If feature scaling is needed, compute scaling statistics using the **training set only**.

For each feature:

$$
x_{\text{scaled}} =
\frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}
$$

where:

- $\mu_{\text{train}}$ = mean of the feature computed from the training set

- $\sigma_{\text{train}}$ = standard deviation of the feature computed from the training set

Then apply the **same** $\mu$ and $\sigma$ values to:
- cross-validation set
- test set

Do **not** compute scaling statistics using the full dataset before splitting.

---

## 8. Why the cross-validation set is better for comparison

If the test set is used to choose between models, it stops being a fair estimate of generalization.

The cross-validation set is the correct place to compare models because:
- it is not used to fit parameters
- it is used only for selection
- the test set remains untouched until the end

---

## 9. Practical workflow

- Split data into train / cv / test
- Fit candidate models on train
- Evaluate on cv
- Choose the best model
- Evaluate once on test
- Report the test result as the final estimate

---

## 10. Key takeaways

- Training error alone is not enough
- Cross-validation is for model selection
- Test error is for final evaluation
- Compute scaling parameters from the training set only
- Keep the test set untouched until the final step

---
## Tags

#neural-networks #deep-learning 