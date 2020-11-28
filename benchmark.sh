#!/usr/bin/bash

python snn_batch_benchmark.py --encoding Poisson --neuron_model IF --update_rule PostPre --n_neurons 100 --batch_size 64 --n_workers 8 --gpu
python snn_batch_benchmark.py --encoding Poisson --neuron_model IF --update_rule WeightDependentPostPre --n_neurons 100 --batch_size 64 --n_workers 8 --gpu
python snn_batch_benchmark.py --encoding Poisson --neuron_model IF --update_rule Hebbian --n_neurons 100 --batch_size 64 --n_workers 8 --gpu
python snn_batch_benchmark.py --encoding Poisson --neuron_model LIF --update_rule PostPre --n_neurons 100 --batch_size 64 --n_workers 8 --gpu
python snn_batch_benchmark.py --encoding Poisson --neuron_model LIF --update_rule WeightDependentPostPre --n_neurons 100 --batch_size 64 --n_workers 8 --gpu
python snn_batch_benchmark.py --encoding Poisson --neuron_model LIF --update_rule Hebbian --n_neurons 100 --batch_size 64 --n_workers 8 --gpu
python snn_batch_benchmark.py --encoding Poisson --neuron_model SRM0 --update_rule PostPre --n_neurons 100 --batch_size 64 --n_workers 8 --gpu
python snn_batch_benchmark.py --encoding Poisson --neuron_model SRM0 --update_rule WeightDependentPostPre --n_neurons 100 --batch_size 64 --n_workers 8 --gpu
python snn_batch_benchmark.py --encoding Poisson --neuron_model SRM0 --update_rule Hebbian --n_neurons 100 --batch_size 64 --n_workers 8 --gpu
python snn_batch_benchmark.py --encoding Poisson --neuron_model DiehlAndCook_Network --update_rule PostPre --n_neurons 100 --batch_size 64 --n_workers 8 --gpu
python snn_batch_benchmark.py --encoding Poisson --neuron_model DiehlAndCook_Network --update_rule WeightDependentPostPre --n_neurons 100 --batch_size 64 --n_workers 8 --gpu
python snn_batch_benchmark.py --encoding Poisson --neuron_model DiehlAndCook_Network --update_rule Hebbian --n_neurons 100 --batch_size 64 --n_workers 8 --gpu
python snn_batch_benchmark.py --encoding Bernoulli --neuron_model IF --update_rule PostPre --n_neurons 100 --batch_size 64 --n_workers 8 --gpu
python snn_batch_benchmark.py --encoding Bernoulli --neuron_model IF --update_rule WeightDependentPostPre --n_neurons 100 --batch_size 64 --n_workers 8 --gpu
python snn_batch_benchmark.py --encoding Bernoulli --neuron_model IF --update_rule Hebbian --n_neurons 100 --batch_size 64 --n_workers 8 --gpu
python snn_batch_benchmark.py --encoding Bernoulli --neuron_model LIF --update_rule PostPre --n_neurons 100 --batch_size 64 --n_workers 8 --gpu
python snn_batch_benchmark.py --encoding Bernoulli --neuron_model LIF --update_rule WeightDependentPostPre --n_neurons 100 --batch_size 64 --n_workers 8 --gpu
python snn_batch_benchmark.py --encoding Bernoulli --neuron_model LIF --update_rule Hebbian --n_neurons 100 --batch_size 64 --n_workers 8 --gpu
python snn_batch_benchmark.py --encoding Bernoulli --neuron_model SRM0 --update_rule PostPre --n_neurons 100 --batch_size 64 --n_workers 8 --gpu
python snn_batch_benchmark.py --encoding Bernoulli --neuron_model SRM0 --update_rule WeightDependentPostPre --n_neurons 100 --batch_size 64 --n_workers 8 --gpu
python snn_batch_benchmark.py --encoding Bernoulli --neuron_model SRM0 --update_rule Hebbian --n_neurons 100 --batch_size 64 --n_workers 8 --gpu
python snn_batch_benchmark.py --encoding Bernoulli --neuron_model DiehlAndCook_Network --update_rule PostPre --n_neurons 100 --batch_size 64 --n_workers 8 --gpu
python snn_batch_benchmark.py --encoding Bernoulli --neuron_model DiehlAndCook_Network --update_rule WeightDependentPostPre --n_neurons 100 --batch_size 64 --n_workers 8 --gpu
python snn_batch_benchmark.py --encoding Bernoulli --neuron_model DiehlAndCook_Network --update_rule Hebbian --n_neurons 100 --batch_size 64 --n_workers 8 --gpu
python snn_batch_benchmark.py --encoding RankOrder --neuron_model IF --update_rule PostPre --n_neurons 100 --batch_size 64 --n_workers 8 --gpu
python snn_batch_benchmark.py --encoding RankOrder --neuron_model IF --update_rule WeightDependentPostPre --n_neurons 100 --batch_size 64 --n_workers 8 --gpu
python snn_batch_benchmark.py --encoding RankOrder --neuron_model IF --update_rule Hebbian --n_neurons 100 --batch_size 64 --n_workers 8 --gpu
python snn_batch_benchmark.py --encoding RankOrder --neuron_model LIF --update_rule PostPre --n_neurons 100 --batch_size 64 --n_workers 8 --gpu
python snn_batch_benchmark.py --encoding RankOrder --neuron_model LIF --update_rule WeightDependentPostPre --n_neurons 100 --batch_size 64 --n_workers 8 --gpu
python snn_batch_benchmark.py --encoding RankOrder --neuron_model LIF --update_rule Hebbian --n_neurons 100 --batch_size 64 --n_workers 8 --gpu
python snn_batch_benchmark.py --encoding RankOrder --neuron_model SRM0 --update_rule PostPre --n_neurons 100 --batch_size 64 --n_workers 8 --gpu
python snn_batch_benchmark.py --encoding RankOrder --neuron_model SRM0 --update_rule WeightDependentPostPre --n_neurons 100 --batch_size 64 --n_workers 8 --gpu
python snn_batch_benchmark.py --encoding RankOrder --neuron_model SRM0 --update_rule Hebbian --n_neurons 100 --batch_size 64 --n_workers 8 --gpu
python snn_batch_benchmark.py --encoding RankOrder --neuron_model DiehlAndCook_Network --update_rule PostPre --n_neurons 100 --batch_size 64 --n_workers 8 --gpu
python snn_batch_benchmark.py --encoding RankOrder --neuron_model DiehlAndCook_Network --update_rule WeightDependentPostPre --n_neurons 100 --batch_size 64 --n_workers 8 --gpu
python snn_batch_benchmark.py --encoding RankOrder --neuron_model DiehlAndCook_Network --update_rule Hebbian --n_neurons 100 --batch_size 64 --n_workers 8 --gpu


