---
layout: default
---
# Methods

In this project, we work with BindsNet framework which uses PyTorch as the backend
library. As everyone knows, Pytorch is a deep learning framework widely used by
researchers on deep learning because it provides facilities to work with GPU.
In this work, we slight modify Diehl and Cook's neural network which has been implemented
in BindsNet and use neuron models as what follows:

1. Integrate and fire model
2. Leaky Integrate and fire model
3. Diehl and Cook model (itself)

Also, we try different learning techniques for each model countered above:

1. Hebbian learning (Similar to STDP except for that the update would be always positive)
2. Weighted STDP (Negative and positive updates are weighted)
3. STDP

And finally, the encodings we employ for converting data into spikes are:

1. Bernoulli (Rate encoding)
2. Poisson (Rate encoding)
3. Rank order (Temporal encding)

Plus, we encode each data pixel in a way that we have a spike sequence of 250 seconds
with the simulation step of 1 second through which we apply encodings mentioned above.

We take the outputs of the model through the excitatory neurons in which each label is
assigned to a group of neurons. For example, if neurons falling in group A fire, then
we take the label of the input as A and so on.


The learning rate set for positive updates of STDP are 0.01 and for negative updates is
1e-36. Also, because of the time-lengthiness of training, we train the model only for 
1 epoch. Additionally, we do mini-batch training with the batch size of 64.
