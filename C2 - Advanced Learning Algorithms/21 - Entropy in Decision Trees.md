## Purpose

Entropy measures the **impurity** of a set of training examples.

- Low entropy → high purity
- High entropy → high impurity

Decision trees use entropy to decide which feature creates the best split.

---

## Class Probabilities

Let:

- `p₁` = fraction of positive examples (e.g., cats)
- `p₀` = fraction of negative examples

$$
p_0 = 1 - p_1
$$

---

## Entropy Formula

$$
H(p_1) = -p_1 \log_2(p_1) - p_0 \log_2(p_0)
$$

Equivalent form:

$$
H(p_1)=
-p_1\log_2(p_1)
-(1-p_1)\log_2(1-p_1)
$$

---

## Key Values

| Positive Fraction (`p₁`) | Entropy |
|---|---:|
| 0.0 | 0 |
| 0.5 | 1 |
| 1.0 | 0 |

Observations:

- Entropy is maximum at a 50-50 split.
- Entropy is minimum when all examples belong to one class.
- More mixed datasets have higher entropy.

---

## Examples

### Perfectly Mixed

3 cats, 3 dogs

$$
p_1 = \frac{3}{6}=0.5
$$

$$
H(p_1)=1
$$

Maximum impurity.

---

### Mostly Cats

5 cats, 1 dog

$$
p_1 = \frac{5}{6}
$$

$$
H(p_1) \approx 0.65
$$

More pure than a 50-50 split.

---

### Mostly Dogs

2 cats, 4 dogs

$$
p_1 = \frac{2}{6}
$$

$$
H(p_1) \approx 0.92
$$

Still fairly impure.

---

### All One Class

6 cats, 0 dogs

or

0 cats, 6 dogs

$$
H(p_1)=0
$$

Completely pure.

---

## Special Case

When:

$$
p_1 = 0
\quad \text{or} \quad
p_0 = 0
$$

the formula contains:

$$
0\log(0)
$$

By convention:

$$
0\log(0)=0
$$

This ensures entropy evaluates correctly to 0.

---

## Why Base-2 Log?

Entropy uses:

$$
\log_2
$$

Using base 2 makes the maximum entropy equal to:

$$
1
$$

Other logarithm bases work too but change the scale.

---

## Entropy and Decision Trees

When choosing a feature to split on:

1. Split the data using a candidate feature.
2. Compute the entropy of the resulting subsets.
3. Prefer splits that produce lower entropy.
4. Lower entropy means higher purity.

Goal:

```text
Choose the feature that creates the purest child nodes.
```

---

## Related Measure: Gini Index

Another impurity measure used in decision trees is:

```text
Gini Impurity
```

Both:
- Entropy
- Gini

usually produce similar results.

This course focuses on entropy.

---

## Key Takeaways

- Entropy measures impurity in a dataset.
- Entropy = 0 → completely pure.
- Entropy = 1 → maximally impure (50-50 split).
- Decision trees choose splits that reduce entropy.
- Lower entropy means better class separation.

---
## Tags

#trees #deep-learning 
