## Motivation

Decision trees can work with continuous-valued features such as:

- Weight
- Height
- Age
- Income

Instead of splitting on discrete values, they split using a threshold.

---

## Threshold-Based Split

Example:

```text
Weight <= 9 ?
```

Creates:

- Left branch: Weight ≤ 9
- Right branch: Weight > 9

---

## Choosing the Best Threshold

For a continuous feature:

1. Try multiple threshold values.
2. Split the data using each threshold.
3. Compute Information Gain.
4. Select the threshold with the highest Information Gain.

---

## Information Gain

$$
IG =
H(p_1^{root})
-
\left(
w^{left}H(p_1^{left})
+
w^{right}H(p_1^{right})
\right)
$$

Choose the split that maximizes Information Gain.

---

## Candidate Thresholds

Common approach:

1. Sort training examples by the feature value.
2. Take midpoints between adjacent values.
3. Evaluate each midpoint.

Example:

```text
Weights:
6, 7, 9, 10, 13

Thresholds:
6.5, 8, 9.5, 11.5
```

For `m` examples, roughly `m - 1` thresholds are tested.

---

## Split Selection Process

For every continuous feature:

```text
For each threshold:
    Compute Information Gain

Choose threshold with highest Information Gain
```

Then compare it against all other candidate features.

Select the feature-threshold pair with the largest Information Gain overall.

---

## After the Split

1. Split the dataset.
2. Create left and right branches.
3. Repeat the decision-tree algorithm recursively.
4. Stop when stopping criteria are met.

---

## Key Takeaways

- Continuous features are handled using threshold-based splits.
- Multiple thresholds are evaluated.
- Information Gain determines the best threshold.
- Continuous and discrete features are treated uniformly during split selection.
- The feature-threshold pair with the highest Information Gain is chosen.

---
## Tags

#trees #deep-learning 