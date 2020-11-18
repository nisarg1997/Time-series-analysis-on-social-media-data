# Topic Modelling

Topic modelling has been conducted for 4 subreddits - Depression, Health Anxiety, Suicide Watch, Addiction -  using an supervised learning algorithm known as Linear Discriminant Analysis.

## Coherence vs Perplexity

Perplexity  is a widely used for language model evaluation. It captures how surprised a model is of new data it has not seen before, and is measured as the normalized log-likelihood of a held-out test set. You can think of the perplexity metric as measuring how probable some new unseen data is given the model that was learned earlier. That is to say, how well does the model represent or reproduce the statistics of the held-out data.

However, recent studies have shown that perplexity and human judgment are often not correlated, and even sometimes slightly anti-correlated.
Optimizing for perplexity may not yield human interpretable topics
This limitation of perplexity measure served as a motivation for more work trying to model the human judgment, and thus Topic Coherence.

Topic Coherence measures score a single topic by measuring the degree of semantic similarity between high scoring words in the topic. These measurements help distinguish between topics that are semantically interpretable topics and topics that are artifacts of statistical inference.

In each of the directories we have Coherence/Perplexity vs Number of Topics

## Algorithm

The Data is first pre-processed. The preprocessing involves:

1. Removal of non-textual characters (eg. emojis)
2. Removal of punctuations
3. Removal of stop words
4. Conversion to Bag of Words Model

Then LDA is performed and its corresponding visualoztions are generated. Different parameters can be tuned here, such as number of topics, to give different LDA visualizations.
The graphs for Perplexity and Coherence vs Number of topics modelled are also generated for further analysis.

## Contents

Each sub directory contains the LDA graphs corresonding to a particular reddit along with possibly the corresponding Coherence and Perplexity graphs. 

1. The ldax.html (where 'x' is a number) is a visualization of LDA for x number of topics.
2. Perplexity.png is a graph of Perplexity vs Number of topics modelled
3. Coherence.png is a graph of Coherence vs Number of topics modelled

## Running the code


```bash
    $ python3 LDA.py
```