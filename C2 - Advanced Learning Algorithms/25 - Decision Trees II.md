# Decision Trees — Complete Notes

## What this covers
- decision-tree learning process
- entropy and information gain
- continuous-valued features
- stopping criteria
- regression trees
- prediction with trained trees

---

## 1. What a decision tree does

A decision tree predicts by following a sequence of feature tests from the root to a leaf.

At each node:
- choose a feature to split on
- split the training examples
- repeat recursively

A leaf node makes the final prediction.

---

## 2. Classification tree workflow

### Training
1. Start with all training examples at the root.
2. Choose the feature with the best split.
3. Split the data into left and right branches.
4. Repeat on each branch.
5. Stop when a stopping rule is met.

### Prediction
For a new example:
1. Start at the root.
2. Follow the test at each node.
3. Move left or right.
4. Reach a leaf.
5. Output the leaf prediction.

---

## 3. Purity and entropy

Decision trees try to create **pure** nodes.

- pure node = mostly or entirely one class
- impure node = mixed classes

Entropy measures impurity.

### Entropy
If `p1` is the fraction of positive examples and `p0 = 1 - p1`:

$$
H(p_1) = -p_1 \log_2(p_1) - p_0 \log_2(p_0)
$$

Properties:
- `H = 0` when the node is pure
- `H = 1` when the node is 50/50
- higher entropy means more impurity

---

## 4. Information gain

To choose a split, compare the impurity before and after splitting.

### Weighted entropy after a split

$$
w^{left}H(p_1^{left}) + w^{right}H(p_1^{right})
$$

### Information gain

$$
IG = H(p_1^{root}) - \Big(w^{left}H(p_1^{left}) + w^{right}H(p_1^{right})\Big)
$$

where:
- `w_left` = fraction of examples going left
- `w_right` = fraction of examples going right

### Rule
Choose the feature/split with the **largest information gain**.

---

## 5. Stopping criteria

Stop splitting when one or more of these happens:

- node is pure
- maximum depth is reached
- information gain is too small
- too few examples remain in the node

Why stop:
- keeps the tree smaller
- reduces overfitting

---

## 6. Continuous features

Decision trees can also use features with numeric values, such as weight.

Instead of splitting on a category, split with a threshold:

$$
\text{weight} \le t ?
$$

### How threshold splits work
1. Sort the training examples by the continuous feature.
2. Try multiple thresholds.
3. Compute information gain for each threshold.
4. Pick the threshold with the highest information gain.

Common choice:
- use midpoints between adjacent sorted values

Then compare that best threshold against all other candidate features.

---

## 7. Regression trees

Decision trees can predict numbers, not just classes.

### Classification tree
- output is a class label

### Regression tree
- output is a real number

At a leaf node in a regression tree:
- predict the **average** of the training targets that reached that leaf

### Split criterion
Instead of entropy, regression trees use **variance reduction**.

#### Variance reduction

$$
Var(root) - \Big(w^{left}Var(left) + w^{right}Var(right)\Big)
$$

Choose the split with the **largest variance reduction**.

---

## 8. Recursive tree building

Decision-tree construction is recursive.

Meaning:
- build the root split
- then build the left subtree
- then build the right subtree
- each subtree is built using the same algorithm

This is why decision-tree implementations often use recursion.

---

## 9. How the tree chooses splits

For each node:
1. test every candidate feature
2. for continuous features, test several thresholds
3. compute entropy reduction or variance reduction
4. choose the best split
5. repeat recursively

Classification trees:
- use entropy / information gain

Regression trees:
- use variance / variance reduction

---

## 10. Practical intuition

### Good split
- creates purer child nodes
- makes predictions easier
- improves the tree

### Bad split
- leaves the data mixed
- gives little gain
- may not be worth doing

---

## 11. Common implementation ideas

In practice, libraries like scikit-learn handle the mechanics of:
- finding the best split
- stopping at depth limits
- growing classification or regression trees

Important hyperparameters often include:
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`

These control complexity and overfitting.

---

## 12. Decision trees in practice

### Strengths
- easy to interpret
- works for classification and regression
- handles categorical and continuous features

### Weaknesses
- can overfit if too deep
- small data changes can change the tree a lot
- greedy splitting may not find globally optimal structure

---

## 13. Quick formulas

### Entropy

$$
H(p_1) = -p_1 \log_2(p_1) - (1-p_1)\log_2(1-p_1)
$$

### Information gain

$$
IG = H(root) - \big(w^{left}H(left) + w^{right}H(right)\big)
$$

### Regression split quality

$$
Var(root) - \big(w^{left}Var(left) + w^{right}Var(right)\big)
$$

---

## 14. One-line summary

A decision tree repeatedly chooses the split that gives the best reduction in impurity, then stops when the tree is pure enough or simple enough.

---
## Tags
#trees 