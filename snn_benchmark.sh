#!/usr/bin/bash

python snn_benchmark.py --encoding Poisson    --neuron_model IF                   --update_rule PostPre 
python snn_benchmark.py --encoding Poisson    --neuron_model IF                   --update_rule WeightDependentPostPre 
python snn_benchmark.py --encoding Poisson    --neuron_model IF                   --update_rule Hebbian 
python snn_benchmark.py --encoding Poisson    --neuron_model LIF                  --update_rule PostPre 
python snn_benchmark.py --encoding Poisson    --neuron_model LIF                  --update_rule WeightDependentPostPre 
python snn_benchmark.py --encoding Poisson    --neuron_model LIF                  --update_rule Hebbian 
python snn_benchmark.py --encoding Poisson    --neuron_model SRM0                 --update_rule PostPre 
python snn_benchmark.py --encoding Poisson    --neuron_model SRM0                 --update_rule WeightDependentPostPre 
python snn_benchmark.py --encoding Poisson    --neuron_model SRM0                 --update_rule Hebbian 
python snn_benchmark.py --encoding Poisson    --neuron_model DiehlAndCook --update_rule PostPre 
python snn_benchmark.py --encoding Poisson    --neuron_model DiehlAndCook --update_rule WeightDependentPostPre 
python snn_benchmark.py --encoding Poisson    --neuron_model DiehlAndCook --update_rule Hebbian 
python snn_benchmark.py --encoding Bernoulli  --neuron_model IF                   --update_rule PostPre 
python snn_benchmark.py --encoding Bernoulli  --neuron_model IF                   --update_rule WeightDependentPostPre 
python snn_benchmark.py --encoding Bernoulli  --neuron_model IF                   --update_rule Hebbian 
python snn_benchmark.py --encoding Bernoulli  --neuron_model LIF                  --update_rule PostPre 
python snn_benchmark.py --encoding Bernoulli  --neuron_model LIF                  --update_rule WeightDependentPostPre 
python snn_benchmark.py --encoding Bernoulli  --neuron_model LIF                  --update_rule Hebbian 
python snn_benchmark.py --encoding Bernoulli  --neuron_model SRM0                 --update_rule PostPre 
python snn_benchmark.py --encoding Bernoulli  --neuron_model SRM0                 --update_rule WeightDependentPostPre 
python snn_benchmark.py --encoding Bernoulli  --neuron_model SRM0                 --update_rule Hebbian 
python snn_benchmark.py --encoding Bernoulli  --neuron_model DiehlAndCook --update_rule PostPre 
python snn_benchmark.py --encoding Bernoulli  --neuron_model DiehlAndCook --update_rule WeightDependentPostPre 
python snn_benchmark.py --encoding Bernoulli  --neuron_model DiehlAndCook --update_rule Hebbian 
python snn_benchmark.py --encoding RankOrder  --neuron_model IF                   --update_rule PostPre 
python snn_benchmark.py --encoding RankOrder  --neuron_model IF                   --update_rule WeightDependentPostPre 
python snn_benchmark.py --encoding RankOrder  --neuron_model IF                   --update_rule Hebbian 
python snn_benchmark.py --encoding RankOrder  --neuron_model LIF                  --update_rule PostPre 
python snn_benchmark.py --encoding RankOrder  --neuron_model LIF                  --update_rule WeightDependentPostPre 
python snn_benchmark.py --encoding RankOrder  --neuron_model LIF                  --update_rule Hebbian 
python snn_benchmark.py --encoding RankOrder  --neuron_model SRM0                 --update_rule PostPre 
python snn_benchmark.py --encoding RankOrder  --neuron_model SRM0                 --update_rule WeightDependentPostPre 
python snn_benchmark.py --encoding RankOrder  --neuron_model SRM0                 --update_rule Hebbian 
python snn_benchmark.py --encoding RankOrder  --neuron_model DiehlAndCook --update_rule PostPre 
python snn_benchmark.py --encoding RankOrder  --neuron_model DiehlAndCook --update_rule WeightDependentPostPre 
python snn_benchmark.py --encoding RankOrder  --neuron_model DiehlAndCook --update_rule Hebbian 


