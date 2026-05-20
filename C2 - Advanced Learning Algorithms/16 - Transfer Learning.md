## Goal
Reuse knowledge from a model trained on a large related dataset to improve performance on a smaller target dataset.

---

## 1. When to use it

Transfer learning is especially useful when:
- you do not have much labeled data
- your target task uses the same input type as the pretraining task
- you want to start from a model that already learned useful features

---

## 2. Basic idea

1. Train a neural network on a large source dataset.
2. Reuse its learned parameters for a new task.
3. Replace the final output layer with one that matches the new task.
4. Fine-tune on your own data.

---

## 3. Example

### Source task
Train on a large image dataset:
- cats
- dogs
- cars
- people
- many classes

### Target task
Recognize handwritten digits:
- 0 through 9

Reuse the earlier layers because they have already learned useful image features.

---

## 4. What gets reused

Usually copy:
- early layers
- middle layers

Usually replace:
- final output layer

The last layer changes because the number of classes is different.

---

## 5. Training options

### Option 1: freeze earlier layers
- keep the pre-trained layers fixed
- train only the new output layer

Useful when:
- the target dataset is very small

### Option 2: fine-tune all layers
- initialize from the pre-trained network
- continue training all layers

Useful when:
- the target dataset is larger
- you want more adaptation

---

## 6. Why it works

Early layers in image models often learn generic features:
- edges
- corners
- curves
- basic shapes

These features transfer well to many vision tasks.

---

## 7. Requirement

The input type must match between pretraining and fine-tuning.

Examples:
- image pretraining → image task
- audio pretraining → audio task
- text pretraining → text task

A model pretrained on images will not usually help much on audio.

---

## 8. Practical benefit

Sometimes you can get strong performance even with a very small target dataset if the pre-trained model is good and the input type matches.

---

## 9. Main takeaway

Transfer learning lets you start from a model trained on a large related dataset, replace the final layer, and fine-tune it on a smaller dataset for your task.

---
## Tags

#model-design 