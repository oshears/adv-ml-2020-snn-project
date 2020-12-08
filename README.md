#  Spiking Neural Networks for Image Classification
## CS 5824/ECE 5424 Project Repository
This repository contains the code used in our CS 5824/ ECE 5424 project: Spiking Neural Networks for Image Classification. The goal of this project was to create several spiking neural network models using [Diehl and Cook's (2015)](https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full) architecure to benchmark their accuracy when classifying images from the MNIST dataset. The models created utilized various encoding schemes, neuron models, and learning rules to perform this task.

## Repository Setup
- Models
- Networks
- 

## Dependencies
Several external packages need to be installed in order for this project to run successfully. Each of these packages is noted below.

### NumPy
[NumPy](https://numpy.org/) is a mature and powerful scientific computing package created by Travis Oliphant in 2005. NumPy enables users to perform complex matrix computations in python and serves as a base for the other packages used in this project.

To install numpy, use the following command:
```pip install numpy```

### PyTorch
[PyTorch](https://pytorch.org/) is a machine learning framework that provides easy to use modules for creating and evaluating artificial neural networks (ANNs).

To install PyTorch, use the following command:
```pip install torch torchvision```

### BindsNET
[BindsNET](https://www.frontiersin.org/articles/10.3389/fninf.2018.00089/full) is a framework developed by Hazan et al. (2018) that provides a streamlined way to construct and evaluate spiking neural networks (SNNs). It uses some of the foundational classes provided in PyTorch in order to provide consistency with the older framework.

```pip install bindsnet```

## Running the SNN Benchmark 
```python benchmark --encoding [Poisson | Bernoulli | RankOrder] --neural_model [IF | LIF | SRM0 | DiehlAndCook] --update_rule [PostPre | WeightDependentPostPre | Hebbian]```

## Running the SNN Benchmark Script
```bash benchmark.sh```

## Running the ANN Benchmark
```python ann_benchmark.py```

## Contributors
- Osaze Shears ([email](oshears@vt.edu))
- Ahmadhossein Yazdani ([email](ahmadyazdani@vt.edu))

## References
> Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Desmaison, A. (2019). Pytorch: An imperative style, high-performance deep learning library. In Advances in neural information processing systems (pp. 8026-8037).

> Hazan, H., Saunders, D. J., Khan, H., Patel, D., Sanghavi, D. T., Siegelmann, H. T., & Kozma, R. (2018). Bindsnet: A machine learning-oriented spiking neural networks library in python. Frontiers in neuroinformatics, 12, 89.
