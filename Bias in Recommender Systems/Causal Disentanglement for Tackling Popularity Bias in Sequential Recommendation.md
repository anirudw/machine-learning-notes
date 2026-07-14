## Abstract
1. Most existing studies tackle this bias in static settings. They neglect the dynamic nature of real-world recommendation scenarios and lack a thorough analysis into the root causes of bias, which makes it challenging to accurately model and mitigate the dynamically changing popularity bias and capture genuine user preferences.
2. Proposed solution - CDSRec (causal disentanglement sequential recommendation model)
3. Uses Markov Chains to model the dynamic nature of bias in recommender systems.
4. Instead of treating item popularity as a single, entirely negative force that needs to be blindly discarded or suppressed, the authors argue that **popularity bias is not always evil**. It has two sides :
	1. Stable intrinsic quality factors (**beneficial bias**): Items that sustain steady, long-term interactions and high ratings. Represented as $y_t$
	2. Dynamic external interference factors(**harmful bias**): Sudden, short-term popularity spikes caused by external factors like viral marketing, social media trends, or celebrity endorsements. Represented as $z_t$
	3. Personalised preferences is represented as $x_t$ and final recommendation score as $S_t$ 
5. We construct a **causal directed acyclic graph** to elucidate the temporal correlations among different factors.
6. In real world the popularity is affected by:
	1. User Interest
	2. Item Popularity
7. **Popularity has both positive and negative impacts.** Empirical analysis on **relation between item popularity and user satisfaction** is done so that the above viewpoint is validated

## Empirical Analysis

1. **Item popularity** is primarily reflected in the frequency of interactions, such as clicks, views, and  purchases.**User satisfaction** can be indicated by the ratings assigned to items, which indirectly reflect user preferences.
2. **Observations:**  Long term popularity implies better user satisfaction (_Perhaps this is the $y_t$_). Short term popularity is not a good indicator of the user satisfaction as it may be because of external factors (_Perhaps the $z_t$ part of the bias_)
3. **Inference:** We classify the latent factors as 3 groups:
	1. **Personalized demand factor ($x_t$):** A match between the item and user preference based on user's history of interaction.
	2. **Inherent quality factors ($y_t$):** Good popularity bias - as caused by long term popularity and high rating.
	3. **External interference factors($z_t$):** Bad popularity bias - as caused by external factors like a viral social media trend or as such. 
4. A solution is modelled so that the first two factors are only considered for the recommendation system. 

## What the model does?

1. $x_t$, $y_t$ and $y_t$ cannot be inferred from raw data. So the authors proposed a way to modify the objective function for neural networks so that **Sequential Variational Autoencoder (VAE)** is optimised for discarding the external $z_t$ factors that it can learn from the temporal latent variables.
2. After empirical analysis with emphasis to temporal aspect, a directed acyclic graph is constructed to elucidate the temporal correlations among different factors

 
## Current Alternatives & Their Drawbacks


1. **Contrastive Learning (CL):** This is a self-supervised learning technique where the model creates slightly modified copies (views) of a user's interaction history. It trains the neural network to pull similar user histories closer together in vector space and push dissimilar ones far away, which helps the model learn cleaner patterns.
2. **Adaptive Augmentation:** The system dynamically alters the training data (e.g., cropping out certain items or masking steps in a sequence). The goal is to **"filter biased samples"** meaning it purposefully modifies or drops interactions that seem heavily influenced by raw popularity rather than genuine user interest.

 _Both seem similar to augmenting images while training. But they are different. Adaptive Augmentation augments the data points **adaptively** ie. when there is a popularity spike, it augments more drastically. However Contrastive Learning is a framework to train such augmented data set - where the augmented images are asked to be learnt as one - ie vectors of the augmented images are pulled together. So To Summarize, AA is a  data preprocessing step while CL is a learning step. **Major Drawback** of these approaches is that these methods still overlook the dynamic nature of real-world scenarios, lacking an in-depth analysis of the root causes of popularity bias and a timely capture of its dynamic changes._ 

### Related Work

The field has evolved from simple probabilistic state transitions (Markov chains) into highly complex, deep topological structures (GNNs and Self-Attention). Each historical architecture has been iteratively optimized to better capture how user preferences organically shift, blend, and evolve over time.


## Output

1. Extensive experimental results on three real-world datasets demonstrate the superiority of our proposed model.