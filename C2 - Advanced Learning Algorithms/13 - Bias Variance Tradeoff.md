## 1. Regularization and model behavior

Regularization controls the trade-off between fitting the training data well and keeping the weights small.

Regularized cost for regression:

$$
J(w,b) = \frac{1}{2m}\sum_{i=1}^{m}\left(f_{w,b}(x^{(i)}) - y^{(i)}\right)^2 + \frac{\lambda}{2m}\sum_{j=1}^{n} w_j^2
$$

- large `λ` → stronger penalty on weights → simpler model
- small `λ` → weaker penalty → more flexible model

### Extreme cases
- `λ = 0`: no regularization, can overfit
- very large `λ`: weights shrink toward 0, model underfits

---

## 2. Effect of λ on bias and variance

- **small λ** → high variance, low training error, higher cross-validation error
- **large λ** → high bias, higher training error, higher cross-validation error
- **intermediate λ** → often best trade-off

### Choosing λ
1. Train models with different `λ` values
2. Compute cross-validation error for each
3. Pick the `λ` with the lowest `J_cv`
4. Report final generalization error using the test set

---

## 3. Train / cross-validation / test split

- **training set**: fit parameters
- **cross-validation set**: choose model or regularization strength
- **test set**: final unbiased evaluation

Do not use the test set to choose `λ` or any other model choice.

---

## 4. Detecting bias and variance

Training error alone is not enough.

A better diagnostic is to compare:
- a baseline level of performance
- training error
- cross-validation error

### High bias
- training error is high relative to the baseline
- cross-validation error is also high
- gap between baseline and training error is large

### High variance
- training error is low
- cross-validation error is much higher than training error
- gap between training and cross-validation error is large

### High bias and high variance
- training error is high relative to baseline
- cross-validation error is much higher than training error

---

## 5. Baseline performance

For some tasks, zero error is unrealistic.

Examples:
- speech recognition
- image understanding
- noisy data problems

A useful baseline can be:
- human-level performance
- a previous strong model
- a known practical target

Use the baseline to judge whether `J_train` is truly high.

---

## 6. Learning curves

Learning curves show error as a function of training set size.

Typical plots:
- `J_train` vs number of training examples
- `J_cv` vs number of training examples

### High bias learning curve
- `J_train` and `J_cv` are both high
- both flatten out
- more data alone usually does not help much

### High variance learning curve
- `J_train` is low
- `J_cv` is much higher
- more training data can help reduce `J_cv`

---

## 7. Learning curve interpretation

### High bias
- model too simple
- adding more data does not fix the main issue
- need a more expressive model

### High variance
- model too flexible
- adding more data often helps
- regularization may also help

---

## 8. Practical workflow

1. Check training and cross-validation errors
2. Compare them to a baseline if available
3. Decide whether the problem is bias, variance, or both
4. Use cross-validation to tune `λ`
5. Use the test set only once at the end

---

## 9. Main takeaways

- `λ` controls the regularization strength
- small `λ` can overfit; large `λ` can underfit
- use cross-validation to choose `λ`
- compare `J_train` and `J_cv` to diagnose bias and variance
- learning curves help reveal whether more data will help
---
## Tags

#model-design 