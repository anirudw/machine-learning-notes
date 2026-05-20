## 1. Scope the project
Define the problem clearly.
- What task are you solving?
- What is the input and output?
- What is the product or user need?

Example:
- speech recognition for voice search

## 2. Collect data
Decide what data is needed.
- gather inputs
- gather labels/transcripts
- make sure the data matches the task

## 3. Train and improve the model
Train an initial model, then iterate.
- train the model
- run error analysis
- run bias/variance analysis
- decide whether to collect more data or change the model

## 4. Iterate
Common loop:
1. train model
2. analyze errors
3. collect more data or improve features
4. retrain

## 5. Deploy
Move the model into production when it is good enough.
- expose the model through an inference server
- let the application call the model via API
- return predictions to the product

## 6. Monitor and maintain
After deployment:
- monitor performance
- log inputs and predictions when allowed
- detect data shift
- retrain or update the model when performance drops

## 7. MLOps
Machine Learning Operations covers:
- building
- deploying
- monitoring
- maintaining
machine learning systems reliably at scale

## Key takeaway
A successful ML project is not just training a model. It is a full cycle:
scope → collect data → train → evaluate → deploy → monitor → improve.

---
## Tags

#model-design 