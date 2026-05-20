## Goal
Improve model performance by increasing or engineering the training data in ways that match the task.

---

## 1. Add more of the right data

Getting more data of everything can be slow and expensive.

A better approach is often to collect more data for the specific error types the model struggles with.

### Workflow
1. Run error analysis.
2. Find the most common error categories.
3. Add more data for those categories.

### Example
If the model fails often on pharmaceutical spam:
- collect more pharmaceutical spam examples
- add features related to drug names or product names

If it fails on phishing:
- collect more phishing examples
- add URL- or routing-based features

---

## 2. Data augmentation

Data augmentation creates new training examples by modifying existing ones while keeping the label the same.

This is especially useful for:
- images
- audio

### Image examples
From one image, create variants by:
- rotating
- scaling up or down
- changing contrast
- applying mild warping

For OCR, these distortions can still represent the same character.

### Audio examples
From one audio clip, create variants by adding:
- crowd noise
- car noise
- phone-channel distortion

The label stays the same, but the training set becomes more robust.

### Good augmentation rule
The transformation should look like the kind of distortion expected in the test set.

Avoid noise that is too random or unrealistic.

---

## 3. Data synthesis

Data synthesis creates brand-new examples from scratch rather than modifying existing ones.

### Example: photo OCR
- generate text using computer fonts
- render it with different colors, contrast, and styles
- use the synthetic images as training data

This can create a very large training set, but realistic synthesis may take significant work.

---

## 4. Data-centric approach

Machine learning can improve in two broad ways:
- model-centric: change the algorithm/model
- data-centric: improve the data

This video emphasizes the data-centric route:
- targeted data collection
- augmentation
- synthesis

---

## 5. Main takeaway

If the model is weak on a specific error type, adding the right kind of data is often more effective than adding more data blindly.

---
## Tags

#model-design 