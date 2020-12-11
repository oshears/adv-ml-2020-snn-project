---
layout: default
---
# Discussion

As explained in the introduction, SNNs are really promising in the sense that they are
really time-efficient as well as being fast. Looks like Diehl and Cook model with poisson
encoding gives the best performance with only one epoch of training.

For most of the deep learning models, hyper parameters like neural network architecture, 
learning rate and batch size and etc. are really effective on 
how the model is trained. For spiking neural networks, we not only do have such concerns
but the input encoding parameters like how much the rate of encoding is, how to choose
the duration of simulation and etc., maybe tuning these parameters can also give us
better performance about the other models other than Diehl and Cook.