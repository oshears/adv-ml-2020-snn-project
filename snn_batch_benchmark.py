import os
import torch
import argparse
import numpy as np

from torchvision import transforms
from tqdm import tqdm

from time import time as t

from bindsnet import ROOT_DIR
from bindsnet.datasets import MNIST, DataLoader
from bindsnet.encoding import PoissonEncoder, BernoulliEncoder, RankOrderEncoder
from bindsnet.learning import PostPre, WeightDependentPostPre, Hebbian
from bindsnet.evaluation import all_activity, proportion_weighting, assign_labels
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import (
    plot_input,
    plot_spikes,
    plot_weights,
    plot_performance,
    plot_assignments,
    plot_voltages,
)

# OYS Added
from models.snn_models import IF_Network, LIF_Network, SRM0_Network, DiehlAndCook_Network, IF_3L_Network, LIF_3L_Network, SRM0_3L_Network, DiehlAndCook_3L_Network

parser = argparse.ArgumentParser()
parser.add_argument("--encoding", type=str, default="Poisson")
parser.add_argument("--neuron_model", type=str, default="IF")
parser.add_argument("--update_rule", type=str, default="PostPre")

args = parser.parse_args()

seed = 0
n_neurons = 100
batch_size = 64
n_epochs = 1
n_test = 10000
n_train = 60000
n_workers = 16
update_steps = 100
exc = 22.5
inh = 120
theta_plus = 0.05
time = 100
dt = 1
intensity = 128
progress_interval = 10
gpu = True

# OYS Added
print("")
print("Encoding Scheme:",args.encoding)
print("Neural Model:",args.neuron_model)
print("Learning Technique:",args.update_rule)
encoding = args.encoding
neuron_model = args.neuron_model
update_rule = None

if args.update_rule == "PostPre":
    update_rule = PostPre
elif args.update_rule == "WeightDependentPostPre":
    update_rule = WeightDependentPostPre
elif args.update_rule == "Hebbian":
    update_rule = Hebbian

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
if n_workers == -1:
    n_workers = gpu * 4 * torch.cuda.device_count()

n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
start_intensity = intensity

# determine update rule




# Build network.
network = None
if neuron_model == "IF":
    network = IF_Network(n_inpt=784,update_rule=update_rule,inpt_shape=(1, 28, 28),batch_size=batch_size,nu=(1e-4, 1e-2))
elif neuron_model == "LIF":
    network = LIF_Network(n_inpt=784,update_rule=update_rule,inpt_shape=(1, 28, 28),batch_size=batch_size,nu=(1e-4, 1e-2))
elif neuron_model == "SRM0":
    network = SRM0_Network(n_inpt=784,update_rule=update_rule,inpt_shape=(1, 28, 28),batch_size=batch_size,nu=(1e-4, 1e-2))
else:
    #network = DiehlAndCook_Network(n_inpt=784,update_rule=update_rule,inpt_shape=(1, 28, 28),batch_size=batch_size,nu=(1e-4, 1e-2))
    from bindsnet.models import DiehlAndCook2015
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

# Directs network to GPU
if gpu:
    network.to("cuda")

# Load MNIST data.
dataset = None
if encoding == "Poisson":
    dataset = MNIST(
        PoissonEncoder(time=time, dt=dt),
        None,
        root=os.path.join(".", "data", "MNIST"),
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
        ),
    )
elif encoding == "Bernoulli":
    dataset = MNIST(
        BernoulliEncoder(time=time, dt=dt),
        None,
        root=os.path.join(ROOT_DIR, "data", "MNIST"),
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
        ),
    )
elif encoding == "RankOrder":
    dataset = MNIST(
        RankOrderEncoder(time=time, dt=dt),
        None,
        root=os.path.join(ROOT_DIR, "data", "MNIST"),
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
        ),
    )


# Neuron assignments and spike proportions.
n_classes = 10
assignments = -torch.ones(n_neurons, device=device)
proportions = torch.zeros((n_neurons, n_classes), device=device)
rates = torch.zeros((n_neurons, n_classes), device=device)

# Sequence of accuracy estimates.
accuracy = {"all": [], "proportion": []}

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

inpt_ims, inpt_axes = None, None
spike_ims, spike_axes = None, None
weights_im = None
assigns_im = None
perf_ax = None
voltage_axes, voltage_ims = None, None

spike_record = torch.zeros((update_interval, int(time / dt), n_neurons), device=device)

# Train the network.
print("\nBegin training.\n")
start = t()

epoch = 0
for epoch in range(n_epochs):
    labels = []

    # Create a dataloader to iterate and batch data
    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=gpu,
    )

    for step, batch in enumerate(train_dataloader):

        if step > n_train:
            break
        
        # Get next input sample.
        inputs = {"X": batch["encoded_image"]}
        if gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}

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

            print("Progress:",step*batch_size,"/",n_train)

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
        #s = spikes["Y"].get("s").permute((1, 0, 2))
        s = spikes["Ae"].get("s").permute((1, 0, 2))
        spike_record[
            (step * batch_size)
            % update_interval : (step * batch_size % update_interval)
            + s.size(0)
        ] = s

        network.reset_state_variables()  # Reset state variables.

print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("Training complete.\n")

# save network
filename = "./networks/snn_" + str(encoding) + "_" + str(neuron_model) + "_" + str(args.update_rule) + ".pt"
network.save(filename)

# Load MNIST data.
test_dataset = None
if encoding == "Poisson":
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
elif encoding == "Bernoulli":
    test_dataset = MNIST(
        BernoulliEncoder(time=time, dt=dt),
        None,
        root=os.path.join(ROOT_DIR, "data", "MNIST"),
        download=True,
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
        ),
    )
elif encoding == "RankOrder":
    test_dataset = MNIST(
        RankOrderEncoder(time=time, dt=dt),
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
    shuffle=False,
    num_workers=32,
    pin_memory=gpu,
)

# Sequence of accuracy estimates.
accuracy = {"all": 0, "proportion": 0}

# Train the network.
print("\nBegin testing\n")
network.train(mode=False)
start = t()

for step, batch in enumerate(test_dataset):
    if step > n_test:
        break
    # Get next input sample.
    inputs = {"X": batch["encoded_image"]}
    if gpu:
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Run the network on the input.
    network.run(inputs=inputs, time=time, input_time_dim=1)

    # Add to spikes recording.
    # spike_record = spikes["Y"].get("s").permute((1, 0, 2))
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

print("\nAll activity accuracy: %.2f" % (accuracy["all"] / n_test))
print("Proportion weighting accuracy: %.2f \n" % (accuracy["proportion"] / n_test))

print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("Testing complete.\n")
