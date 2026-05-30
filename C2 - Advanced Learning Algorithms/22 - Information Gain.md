# Information Gain and Decision Tree Construction

## Information Gain

Decision trees choose the feature that produces the **largest reduction in entropy**.

### Goal
Choose the split that creates the purest child nodes.

---

## Weighted Entropy After Split

For a candidate split:

$$
\text{Weighted Entropy}
=
w^{left}H(p_1^{left})
+
w^{right}H(p_1^{right})
$$

where:

- $w^{left}$ = fraction of examples sent to the left branch
- $w^{right}$ = fraction of examples sent to the right branch
- $p_1^{left}$ = fraction of positive examples in the left branch
- $p_1^{right}$ = fraction of positive examples in the right branch

---

## Information Gain Formula

$$
IG =
H(p_1^{root})
-
\Big(
w^{left}H(p_1^{left})
+
w^{right}H(p_1^{right})
\Big)
$$

where:

- $H(p_1^{root})$ = entropy before splitting
- weighted entropy = entropy after splitting

### Interpretation

$$
\text{Information Gain}
=
\text{Entropy Before Split}
-
\text{Entropy After Split}
$$

Higher information gain = better split.

---

## Choosing a Split

For every candidate feature:

1. Compute entropy of left branch
2. Compute entropy of right branch
3. Compute weighted entropy
4. Compute information gain
5. Select the feature with the highest information gain

---

## Why Information Gain Works

A good split:

- separates classes well
- creates purer child nodes
- reduces entropy significantly

A poor split:

- leaves classes mixed
- produces little reduction in entropy

---

# Decision Tree Learning Algorithm

## Step 1: Start at Root

Place all training examples at the root node.

---

## Step 2: Compute Information Gain

For every available feature:

$$
IG(feature)
$$

Compute information gain.

Choose:

$$
\arg\max IG
$$

---

## Step 3: Split Data

Split examples according to the selected feature.

Create:
- left branch
- right branch

Send examples to the appropriate branch.

---

## Step 4: Repeat Recursively

For each child node:

1. Check stopping criteria
2. If not satisfied:
   - compute information gain again
   - choose best feature
   - split again

Repeat until stopping criteria are met.

---

# Stopping Criteria

Stop splitting when one or more conditions hold.

## 1. Pure Node

Entropy is zero.

All examples belong to the same class.

```text
All Cats
or
All Dogs
```

Create a leaf node.

---

## 2. Maximum Depth Reached

Tree depth exceeds a predefined limit.

Benefits:
- smaller tree
- less overfitting

---

## 3. Information Gain Too Small

Stop if:

$$
IG < \epsilon
$$

for some threshold $\epsilon$.

Reason:
- split provides little benefit
- avoids unnecessary complexity

---

## 4. Too Few Examples

Stop if the node contains very few training examples.

Reason:
- prevents overfitting
- avoids unreliable splits

---

# Recursive Nature of Decision Trees

Decision tree construction is recursive.

A large tree is built by repeatedly building smaller subtrees.

```text
Root
├── Left Subtree
└── Right Subtree
```

Each subtree is itself constructed using the same algorithm.

---

# Model Complexity

Increasing maximum depth:

- increases model complexity
- increases expressive power
- increases risk of overfitting

Decreasing maximum depth:

- simpler model
- less overfitting
- may underfit

Maximum depth behaves similarly to:
- polynomial degree
- neural network size

---

# Prediction with a Decision Tree

For a new example:

1. Start at the root
2. Evaluate the split condition
3. Move left or right
4. Repeat until reaching a leaf node
5. Output the leaf-node prediction

---

# Key Takeaways

- Entropy measures impurity.
- Information gain measures reduction in entropy.
- Decision trees choose the feature with the highest information gain.
- Splitting continues recursively.
- Stopping criteria help prevent overfitting.
- Prediction is performed by traversing the tree from root to leaf.

---
## Tags
#trees #deep-learning 
