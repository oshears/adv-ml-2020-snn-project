---
layout: default
---
# Discussion

As explained in the introduction, SNNs are really promising in the sense that they are really time-efficient as well as being fast. The Diehl and Cook model with poisson encoding gives the best performance with only one epoch of training. The reason for this is beacuse the hyperparameters used in this project are likely adjusted to optimize the performance of the adaptive threshold neurons, while the other neuron models fail to reach the same level of performance.

For most of the deep learning models, hyper parameters like neural network architecture, 
learning rate, batch size and etc. are really effective on 
how the model is trained. For spiking neural networks, we not only have such concerns,
but also concerns about the input encoding parameters (i.e., how much the rate of encoding is), how to choose
the duration of simulation and etc. Tuning these parameters may also give us
better performance when compared to other models aside from Diehl and Cook's.