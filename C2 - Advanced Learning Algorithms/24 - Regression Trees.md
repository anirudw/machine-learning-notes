## Goal

Use decision trees for **regression problems** where the target is a continuous value.

Examples:
- Weight prediction
- House-price prediction
- Sales forecasting

---

## Key Difference from Classification Trees

### Classification Tree
Predicts a class label.

Leaf nodes contain:

```text
Cat
Dog
Spam
Not Spam
```

Split criterion:
- Entropy
- Information Gain

---

### Regression Tree
Predicts a numeric value.

Leaf nodes contain:

```text
8.35
17.70
9.90
```

Split criterion:
- Variance Reduction

---

## Prediction at a Leaf Node

A regression tree predicts the **average target value** of the training examples that reach that leaf.

Example:

Training values at leaf:

```text
7.2, 7.6, 8.4, 10.2
```

Prediction:

$$
\hat{y}
=
\frac{7.2 + 7.6 + 8.4 + 10.2}{4}
=
8.35
$$

---

## Variance

Variance measures how spread out the target values are.

- Low variance → values are similar
- High variance → values vary widely

Goal:

```text
Create child nodes with low variance.
```

---

## Weighted Variance After a Split

$$
W_{left}\,Var(left)
+
W_{right}\,Var(right)
$$

where:

- $W_{left}$ = fraction of examples sent left
- $W_{right}$ = fraction of examples sent right

---

## Variance Reduction

Regression trees choose splits that maximize variance reduction.

$$
	ext{Variance Reduction}
=
Var(root)
-
\Big(
W_{left}Var(left)
+
W_{right}Var(right)
\Big)
$$

Interpretation:

```text
Variance Before Split
-
Variance After Split
```

Higher variance reduction = better split.

---

## Split Selection Algorithm

For each feature:

1. Split the data.
2. Compute child-node variances.
3. Compute weighted variance.
4. Compute variance reduction.

Choose:

```text
Feature with maximum variance reduction.
```

---

## Building the Tree

1. Start at the root node.
2. Select the feature with the largest variance reduction.
3. Split the data.
4. Repeat recursively on each branch.
5. Stop when stopping criteria are met.

Typical stopping criteria:

- maximum depth reached
- too few examples
- variance reduction below threshold

---

## Comparison

| Classification Tree | Regression Tree |
|----------|----------|
| Predicts class | Predicts number |
| Uses entropy | Uses variance |
| Uses information gain | Uses variance reduction |
| Leaf = class prediction | Leaf = average target value |

---

## Key Takeaways

- Regression trees predict continuous values.
- Leaf nodes output the average target value.
- Splits are chosen using variance reduction.
- The feature with the highest variance reduction is selected.
- Tree construction remains recursive, just like classification trees.

---
## Tags

#trees #deep-learning 