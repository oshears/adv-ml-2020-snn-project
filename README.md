#  Spiking Neural Networks for Image Classification
## CS 5824/ECE 5424 Project Repository
This repository contains the code used in our CS 5824/ ECE 5424 project.

## Repository Setup
- Models
- Networks
- 

## Dependencies

### NumPy
```pip install numpy```

### PyTorch
```pip install torch torchvision```

### BindsNET
```pip install bindsnet```

## Running the SNN Benchmark 
```python benchmark --encoding [Poisson | Bernoulli | RankOrder] --neural_model [IF | LIF | SRM0 | DiehlAndCook] --update_rule [PostPre | WeightDependentPostPre | Hebbian]```

## Running the SNN Benchmark Script
```bash benchmark.sh```

## Running the ANN Benchmark
```python ann_benchmark.py```
