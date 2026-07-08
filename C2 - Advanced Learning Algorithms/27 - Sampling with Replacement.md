## What it means
Sampling with replacement means:

1. pick one example
2. put it back
3. pick again

The same example can appear more than once.

---

## Why it matters for tree ensembles
Sampling with replacement is used to create new training sets that are:
- similar to the original set
- but slightly different from each other

These different samples help produce different trees.

---

## Example behavior
If you sample 4 times from 4 items with replacement, you might get:
- green, yellow, blue, blue

Some items may repeat, and some may not appear at all.

---

## In decision-tree ensembles
Each tree is trained on a different bootstrap sample of the original dataset.

This helps the trees vary, which improves the overall ensemble.

---

## Key takeaway
Sampling with replacement is the basic trick used to build different training sets for different trees.


---
## Tags
#trees #deep-learning 