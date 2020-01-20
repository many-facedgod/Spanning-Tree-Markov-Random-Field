# Spanning-tree Approximations for Markov Random Fields

**Work in progress**

The repository has a couple of models that are spanning tree approximations (for which we can calculate the normalization constant using dynamic programming) for grid-like undirected models over binary matrices like images. 

The first idea is to learn a weight for each edge along with the other parameters, such that the spanning tree used is the maximum spanning tree given weights. Of course, since this operation is not differentiable, we can use REINFORCE[1] to try to learn a distribution over the weights, trying to maximizing an ELBO. This does not work very well because the stochastic gradient is too noisy given the high dimensionality of the weight vector.

The second idea is to use an ensemble of spanning tree models, which each spanning tree randomly generated and fixed during initialization (similar to an idea proposed in [2]).

The goal is to use this as a final layer of deep neural network for image segmentation such that the "parameters" are the output of the network, and then the model produces a conditional distribution over the possible segmentations of the image.

## Requirements
- numpy
- PyTorch
- tqdm
- matplotlib
- Python 3

## Implementation

The two kinds of generative models have been implemented, and an example over the binarized MNIST dataset is shown in `mnist_gen_classifier.py`, which trains a generative model over each digit and then uses the generative models for classification. The classifier also allows simultaneous discriminative training for the different models.

The model type can be changed in the `params` dictionary in the file. The models also support Gibbs sampling. `model.eval()` should be called before any evaluation/sampling which caches the normalization constant to avoid repeated computations. Similarly, `model.train()` should be called before any training.


## References
1. Sutton, Richard S., et al. "Policy gradient methods for reinforcement learning with function approximation." Advances in neural information processing systems. 2000.
2. Pletscher, Patrick, Cheng Soon Ong, and Joachim Buhmann. "Spanning tree approximations for conditional random fields." Artificial Intelligence and Statistics. 2009.