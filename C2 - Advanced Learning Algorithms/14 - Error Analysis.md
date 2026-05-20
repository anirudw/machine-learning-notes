# Error Analysis

## Goal
Use a small sample of misclassified examples to decide what to improve next.

---

## 1. What error analysis is

Error analysis means manually inspecting misclassified examples from the cross-validation set to understand where the model is failing.

Typical process:
1. Collect misclassified examples.
2. Group them by common type or pattern.
3. Count how often each type appears.
4. Use the counts to decide what improvement is most promising.

---

## 2. Example workflow

If the model misclassifies 100 cross-validation examples, inspect those 100 and classify them by error type.

Example categories:
- pharmaceutical spam
- deliberate misspellings
- unusual email routing
- phishing / password-stealing emails
- image-based spam

These categories can overlap. One example may belong to multiple categories.

---

## 3. How to use the results

Use the counts to decide what to do next.

Examples:
- If pharmaceutical spam is common, collect more pharma-spam examples or add features related to drug names.
- If phishing is common, add URL-based or routing-based features.
- If misspellings are rare, spending a lot of time on them may not help much.

The point is to focus effort on the biggest sources of error.

---

## 4. How many examples to inspect

If the cross-validation set is large, inspect a random sample of about 100 to a few hundred misclassified examples.

That is usually enough to reveal the main error patterns without manually reviewing everything.

---

## 5. When error analysis is useful

Error analysis is especially helpful when:
- humans can judge the errors reasonably well
- the task has understandable failure modes
- you need to choose between possible improvements

It is harder for tasks where humans themselves are not good at the task.

---

## 6. Relation to bias/variance

Bias/variance analysis tells you whether more data is likely to help.  
Error analysis tells you what kind of improvement is most promising.

Together, they help guide the next change to the model.

---

## 7. Main takeaway

Error analysis is a manual but powerful diagnostic tool.  
It helps you prioritize the most useful fixes instead of wasting time on low-impact changes.

---
## Tags

#model-design 