import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms
from tqdm import tqdm

from time import time as t

from bindsnet import ROOT_DIR
from bindsnet.datasets import MNIST, DataLoader
from bindsnet.encoding import PoissonEncoder
from bindsnet.evaluation import all_activity, proportion_weighting, assign_labels
from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor
from bindsnet.network.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.learning import PostPre
from bindsnet.utils import get_square_weights, get_square_assignments
from bindsnet.analysis.plotting import (
    plot_input,
    plot_spikes,
    plot_weights,
    plot_performance,
    plot_assignments,
    plot_voltages,
)

seed = 0
n_neurons = 100
n_train = 60000
n_test = 10000
exc = 22.5
inh = 120
theta_plus = 0.05
time = 250
dt = 1.0
intensity = 32
progress_interval = 10
update_interval = 250
gpu = True
n_epochs = 1
update_steps = 256
batch_size = 64

update_interval = update_steps * batch_size



device = "cpu"
# Sets up Gpu use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if gpu and torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)
    device = "cpu"
    if gpu:
        gpu = False

torch.set_num_threads(os.cpu_count() - 1)
print("Running on Device = ", device)

# Determines number of workers to use
n_workers = 16

n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
start_intensity = intensity

# Build network.
network = DiehlAndCook2015(
    n_inpt=784,
    n_neurons=n_neurons,
    exc=exc,
    inh=inh,
    dt=dt,
    norm=78.4,
    nu=(1e-4, 1e-2),
    theta_plus=theta_plus,
    inpt_shape=(1, 28, 28),
)

network.to(device)

# Load MNIST data.
dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join("data"),
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

# Create a dataloader to iterate and batch data
train_dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=n_workers,
)

exc_voltage_monitor = Monitor(network.layers["Ae"], ["v"], time=int(time / dt))
network.add_monitor(exc_voltage_monitor, name="exc_voltage")

# Set up monitors for spikes and voltages
spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(
        network.layers[layer], state_vars=["s"], time=int(time / dt)
    )
    network.add_monitor(spikes[layer], name="%s_spikes" % layer)

voltages = {}
for layer in set(network.layers) - {"X"}:
    voltages[layer] = Monitor(
        network.layers[layer], state_vars=["v"], time=int(time / dt)
    )
    network.add_monitor(voltages[layer], name="%s_voltages" % layer)

    # Neuron assignments and spike proportions.
n_classes = 10

def fit(network, train_dataloader, n_neurons,\
        n_classes, device, spikes, update_interval, update_steps\
        , time, dt, n_epochs=1, batch_size=64):
  assignments = -torch.ones(n_neurons, device=device)
  proportions = torch.zeros((n_neurons, n_classes), device=device)
  rates = torch.zeros((n_neurons, n_classes), device=device)

  # Sequence of accuracy estimates.
  accuracy = {"all": [], "proportion": []}

  # Voltage recording for excitatory and inhibitory layers.

  spike_record = torch.zeros((update_interval, int(time / dt), n_neurons), device=device)

  # Train the network.
  print("\nBegin training.\n")
  start = t()

  for epoch in range(n_epochs):
      labels = []

      if epoch % progress_interval == 0:
          print("Progress: %d / %d (%.4f seconds)" % (epoch, n_epochs, t() - start))
          start = t()

      pbar = tqdm(total=len(train_dataloader), position=0, leave=True)

      for step, batch in enumerate(train_dataloader):
          if step > n_train:
              break
          # Get next input sample.
          inputs = {"X": batch["encoded_image"]}
          inputs = {k: v.to(device) for k, v in inputs.items()}

          if step % update_steps == 0 and step > 0:
              # Convert the array of labels into a tensor
              label_tensor = torch.tensor(labels, device=device)

              # Get network predictions.
              all_activity_pred = all_activity(
                  spikes=spike_record, assignments=assignments, n_labels=n_classes
              )
              proportion_pred = proportion_weighting(
                  spikes=spike_record,
                  assignments=assignments,
                  proportions=proportions,
                  n_labels=n_classes,
              )

              # Compute network accuracy according to available classification strategies.
              accuracy["all"].append(
                  100
                  * torch.sum(label_tensor.long() == all_activity_pred).item()
                  / len(label_tensor)
              )
              accuracy["proportion"].append(
                  100
                  * torch.sum(label_tensor.long() == proportion_pred).item()
                  / len(label_tensor)
              )

              print(
                  "\nAll activity accuracy: %.2f (last), %.2f (average), %.2f (best)"
                  % (
                      accuracy["all"][-1],
                      np.mean(accuracy["all"]),
                      np.max(accuracy["all"]),
                  )
              )
              print(
                  "Proportion weighting accuracy: %.2f (last), %.2f (average), %.2f"
                  " (best)\n"
                  % (
                      accuracy["proportion"][-1],
                      np.mean(accuracy["proportion"]),
                      np.max(accuracy["proportion"]),
                  )
              )

              # Assign labels to excitatory layer neurons.
              assignments, proportions, rates = assign_labels(
                  spikes=spike_record,
                  labels=label_tensor,
                  n_labels=n_classes,
                  rates=rates,
              )

              labels = []

          labels.extend(batch["label"].tolist())

          # Run the network on the input.
          network.run(inputs=inputs, time=time, input_time_dim=1)

          # Add to spikes recording.
          s = spikes["Ae"].get("s").permute((1, 0, 2))
          spike_record[
              (step * batch_size)
              % update_interval : (step * batch_size % update_interval)
              + s.size(0)
          ] = s


          network.reset_state_variables()  # Reset state variables.

          pbar.update()


  print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
  print("Training complete.\n")
  return assignments, proportions

assignments, proportions = fit(network, train_dataloader, n_neurons, n_classes, device, spikes, update_interval,\
                               update_steps, time, dt)

# Load MNIST data.
test_dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join(ROOT_DIR, "data", "MNIST"),
    download=True,
    train=False,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

# Create a dataloader to iterate and batch data
test_dataloader = DataLoader(
    test_dataset,
    batch_size=256,
    num_workers=32,
    shuffle=False,
)

def test(network, test_dataloader, device, spikes, assignments, proportions):
  # Sequence of accuracy estimates.
  accuracy = {"all": 0, "proportion": 0}

  # Train the network.
  print("\nBegin testing\n")
  network.train(mode=False)
  start = t()

  n_test = len(test_dataloader.dataset)
  pbar = tqdm(total=n_test, position=0, leave=True)
  for step, batch in enumerate(test_dataloader):
      if step > n_test:
          break
      # Get next input sample.
      inputs = {"X": batch["encoded_image"]}
      inputs = {k: v.to(device) for k, v in inputs.items()}

      # Run the network on the input.
      network.run(inputs=inputs, time=time, input_time_dim=1)

      # Add to spikes recording.
      spike_record = spikes["Ae"].get("s").permute((1, 0, 2))

      # Convert the array of labels into a tensor
      label_tensor = torch.tensor(batch["label"], device=device)

      # Get network predictions.
      all_activity_pred = all_activity(
          spikes=spike_record, assignments=assignments, n_labels=n_classes
      )
      proportion_pred = proportion_weighting(
          spikes=spike_record,
          assignments=assignments,
          proportions=proportions,
          n_labels=n_classes,
      )

      # Compute network accuracy according to available classification strategies.
      accuracy["all"] += float(torch.sum(label_tensor.long() == all_activity_pred).item())
      accuracy["proportion"] += float(
          torch.sum(label_tensor.long() == proportion_pred).item()
      )

      network.reset_state_variables()  # Reset state variables.
      pbar.set_description_str("Test progress: ")
      pbar.update()
      
  print("\nAll activity accuracy: %.2f" % (accuracy["all"] * 100 / n_test))
  print("Proportion weighting accuracy: %.2f \n" % (accuracy["proportion"] * 100/ n_test))

  print("Testing complete.\n")
  
test(network, test_dataloader, device, spikes, assignments, proportions)
