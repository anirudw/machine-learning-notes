## Why not a single tree?
A single decision tree can be sensitive to small changes in the training set.

A small change in one training example can change:
- the best split at the root
- the structure of the entire tree
- the final prediction path

This makes one tree less robust.

---

## Idea of an ensemble
Instead of one tree, build many trees and let them vote.

For classification:
- each tree gives a prediction
- the ensemble uses majority vote

This usually gives better performance and more stable predictions.

---

## Main benefit
A tree ensemble is less sensitive to small changes in the data because the final prediction is based on many trees, not just one.

---

## Key takeaway
A tree ensemble improves robustness and accuracy by averaging or voting over many decision trees.

---
## Tags

#trees #deep-learning 
