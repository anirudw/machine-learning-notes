
## Introduction

1. CF-based algorithms grapple with inherent challenges, notably the cold-start problem, where limited data on new users or items hinders accurate recommendations, and data sparsity, which stems from the generally sparse interaction data in large-scale datasets
2. Popularity bias: 
	1. popular items keep getting recommended, while niche or less-known items are ignored.
	2. Popularity bias also leads to calibration issues, where recommendations may not match individual user tastes for popular or niche content, reducing personalization
	3. Niche-focused users, in particular, suffer the most because they receive fewer recommendations that match their preferences, while blockbuster-focused users experience less harm since they naturally enjoy popular items.
3. Many studies just focus on static analysis of bias - ie, a one-shot approach (immediate effect of the recommendation is only studied), but in practice the bias propagation happens through feedback loops.
4. In practice, RSs run in dynamic settings, where user preferences change over multiple recommendation cycles.
	1. recommendations influence user actions, which then shape how the recommendation model learns. 
	2. Although real-time feedback loops could offer deeper insights, offline experiments usually cannot collect user feedback for each cycle because they do not have access to live user interactions, something that only platforms can do.

## Framework proposed

1. A feedback loop framework to explore how recommendation quality changes over time in several areas including accuracy, beyond-accuracy, fairness, and calibration in terms of popularity for users with different levels of interest in popular items.
2. In each cycle, our framework generates synthetic ratings for users based on two explicitly separated scenarios: 
	1. **Repeat consumption** preserves a user’s historical ratings while adding user-specific Gaussian noise to capture natural preference variability.
	2. **New consumption** predictions are produced via a scale-consistent item-based CF method with controlled noise. This design ensures realistic simulation of user behavior and preserves the original rating distribution across cycles.