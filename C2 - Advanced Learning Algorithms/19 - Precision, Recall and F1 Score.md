## Why accuracy can fail
For skewed data, accuracy can be misleading.

Example:
- rare disease detection
- only 0.5% of patients have the disease

A model that predicts `0` every time gets:
- 99.5% accuracy
- but is useless, because it never detects the disease

So classification error / accuracy is not enough when positive and negative classes are very imbalanced.

---

## Confusion matrix

|  | Actual 1 | Actual 0 |
|---|---:|---:|
| Predicted 1 | True Positive (TP) | False Positive (FP) |
| Predicted 0 | False Negative (FN) | True Negative (TN) |

Meanings:
- **TP**: predicted positive and was actually positive
- **FP**: predicted positive but was actually negative
- **FN**: predicted negative but was actually positive
- **TN**: predicted negative and was actually negative

---

## Precision

Precision measures how accurate positive predictions are.

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

Interpretation:
- Of all examples predicted positive, how many were actually positive?

Example:
- `TP = 15`
- `FP = 5`

$$
\text{Precision} = \frac{15}{15+5} = 0.75
$$

---

## Recall

Recall measures how many actual positives were found.

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

Interpretation:
- Of all actual positive examples, how many did the model correctly detect?

Example:
- `TP = 15`
- `FN = 10`

$$
\text{Recall} = \frac{15}{15+10} = 0.60
$$

---

## Why precision and recall matter
For skewed classes:
- accuracy can look good even when the model is useless
- precision and recall reveal whether the model is actually detecting the rare class

If the model predicts `0` all the time:
- precision is effectively `0`
- recall is `0`

---

## Precision–recall trade-off

Logistic regression outputs probabilities in `[0, 1]`.

A decision threshold is usually applied:

- predict `1` if `f(x) >= threshold`
- predict `0` otherwise

### Higher threshold
Example: `0.7` or `0.9`

- fewer positive predictions
- **higher precision**
- **lower recall**

### Lower threshold
Example: `0.3`

- more positive predictions
- **lower precision**
- **higher recall**

So precision and recall usually trade off against each other.

---

## F1 score

If you want a single number to combine precision and recall, use the F1 score.

$$
F_1 = \frac{2PR}{P+R}
$$

where:
- `P` = precision
- `R` = recall

Why F1 is useful:
- it gives more weight to the smaller of precision and recall
- it is better than taking the plain average when one metric is very low

---

## When to use what

- **Accuracy**: only when classes are reasonably balanced
- **Precision**: when false positives are especially costly
- **Recall**: when false negatives are especially costly
- **F1 score**: when you want one metric that balances precision and recall

---

## Key takeaway
For skewed data sets, do not rely on accuracy alone. Use the confusion matrix, then precision and recall, and use F1 if you need a single summary metric.

---
## Tags
#model-design #deep-learning 