## Goal
Build a decision tree by repeatedly splitting the training set into purer subsets.

## Basic workflow
1. Choose a feature for the root node.
2. Split the data into subsets based on that feature.
3. For each branch, choose the next feature to split on.
4. Continue until the nodes become pure or a stopping condition is met.

## Example structure
- Root split: ear shape
- Left branch: face shape
- Right branch: whiskers

Each split should move the examples closer to a single class, such as:
- all cats
- all dogs

## Why purity matters
A good split creates subsets that are as pure as possible.

A feature is useful if it separates the labels well.  
For example, if a feature perfectly separates cats from dogs, it would create very pure branches.

## When to stop splitting
Common stopping rules:
- node is already pure
- maximum depth is reached
- improvement in purity is too small
- number of examples in the node is too small

## Depth
- Root node has depth 0
- Its children have depth 1
- Depth increases by 1 for each step away from the root

Limiting depth helps:
- keep the tree smaller
- reduce overfitting

## Why the algorithm can feel messy
Decision trees have many design choices:
- which feature to split on
- when to stop splitting
- how deep the tree can grow

These are refinements added over time, but they fit together into a practical learning algorithm.

## Main takeaway
Decision tree learning is about choosing splits that increase purity and stopping before the tree becomes too large or overfit.

---

## Tags

#trees #deep-learning 