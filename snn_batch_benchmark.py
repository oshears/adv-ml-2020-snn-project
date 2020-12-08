
# import numpy
import numpy as np

# import modules from pytorch
import torch
from torchvision import transforms

# import modules from bindsnet
from bindsnet.datasets import MNIST, DataLoader
from bindsnet.encoding import PoissonEncoder, BernoulliEncoder, RankOrderEncoder
from bindsnet.learning import PostPre, WeightDependentPostPre, Hebbian
from bindsnet.evaluation import all_activity, proportion_weighting, assign_labels
from bindsnet.network.monitors import Monitor

# miscellaneous imports
import os
import argparse


# import local modules
from models.snn_models import IF_Network, LIF_Network, SRM0_Network, DiehlAndCook_Network

# create an argument parser to interpret command line arguments
parser = argparse.ArgumentParser()

# --encoding specifies the type of encoding (Poisson, Bernoulli or RankOrder)
parser.add_argument("--encoding", type=str, default="Poisson")

# --neuron_model specifies the type of neural model (IF, LIF, SRM0, or DiehlAndCook (Adaptive))
parser.add_argument("--neuron_model", type=str, default="IF")

# -- update_rule specifies the type of learning rule (PostPre, WeightDependentPostPre, or Hebbian)
parser.add_argument("--update_rule", type=str, default="PostPre")

# parse the arguments
args = parser.parse_args()

# declare global variables

# n_neurons specifies the number of neurons per layer
n_neurons = 100

# batch_size specifies the number of training samples to collect weight changes from before updating the weights
batch_size = 64

# n_train specifies the number of training samples
n_train = 60000

# n_test specifies the number of testing samples
n_test = 10000

# update_steps specifies the number of batches to process before reporting an update
update_steps = 100

# time specifies the simulation time of the SNN
time = 100

# dt specifies the timestep size for the simulation time
dt = 1

# intensity specifies the maximum intensity of the input data
intensity = 128

# report the selected encoding scheme, neural model and learning technique
print("Encoding Scheme:",args.encoding)
print("Neural Model:",args.neuron_model)
print("Learning Technique:",args.update_rule)

# assign a value to the encoder based on the input argument
encoder = None
if args.encoding == "Poisson":
    encoder = PoissonEncoder(time=time,dt=dt)
if args.encoding == "Bernoulli":
    encoder = BernoulliEncoder(time=time,dt=dt)
if args.encoding == "RankOrder":
    encoder = RankOrderEncoder(time=time,dt=dt)

# assign a value to the update_rule based on the input argument
update_rule = None
if args.update_rule == "PostPre":
    update_rule = PostPre
elif args.update_rule == "WeightDependentPostPre":
    update_rule = WeightDependentPostPre
elif args.update_rule == "Hebbian":
    update_rule = Hebbian

update_interval = update_steps * batch_size

# Sets up Gpu use
device = torch.device("cuda")
#torch.cuda.manual_seed_all(seed)
torch.set_num_threads(os.cpu_count() - 1)
print("Running on Device = ", device)

# Determines number of workers to use
n_workers = 4 * torch.cuda.device_count()


# Build network.
network = None
if args.neuron_model == "IF":
    network = IF_Network(n_inputs=784,update_rule=update_rule,input_shape=(1, 28, 28),batch_size=batch_size)
elif args.neuron_model == "LIF":
    network = LIF_Network(n_inputs=784,update_rule=update_rule,input_shape=(1, 28, 28),batch_size=batch_size)
elif args.neuron_model == "SRM0":
    network = SRM0_Network(n_inputs=784,update_rule=update_rule,input_shape=(1, 28, 28),batch_size=batch_size)
elif args.neuron_model == "DiehlAndCook":
    network = DiehlAndCook_Network(n_inputs=784,update_rule=update_rule,input_shape=(1, 28, 28),batch_size=batch_size)

# Directs network to GPU
network.to("cuda")

# Load MNIST data.
dataset = MNIST(
    encoder,
    None,
    root=os.path.join(".", "data", "MNIST"),
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

# Set up monitors for spikes
output_spikes_monitor = Monitor(network.layers["Y"], state_vars=["s"], time=int(time / dt))
network.add_monitor(output_spikes_monitor, name="Y")
spike_record = torch.zeros((update_interval, int(time / dt), n_neurons), device=device)

# Train the network.
print("\nBegin training.\n")

labels = []

# Create a dataloader to iterate and batch data
train_dataloader = DataLoader( dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=True, )

for step, batch in enumerate(train_dataloader):

    # Get next input sample.
    #inputs = {"X": batch["encoded_image"]}
    inputs = {"X": batch["encoded_image"].cuda()}

    if step % update_steps == 0 and step > 0:
        # Convert the array of labels into a tensor
        label_tensor = torch.tensor(labels, device=device)

        # Get network predictions.
        all_activity_pred = all_activity( spikes=spike_record, assignments=assignments, n_labels=n_classes )
        proportion_pred = proportion_weighting( spikes=spike_record, assignments=assignments, proportions=proportions, n_labels=n_classes, )

        # Compute network accuracy according to available classification strategies.
        accuracy["all"].append( 100 * torch.sum(label_tensor.long() == all_activity_pred).item() / len(label_tensor) )
        accuracy["proportion"].append( 100 * torch.sum(label_tensor.long() == proportion_pred).item() / len(label_tensor) )

        print( "\nAll activity accuracy: %.2f (last), %.2f (average), %.2f (best)" % ( accuracy["all"][-1], np.mean(accuracy["all"]), np.max(accuracy["all"]), ) )
        print("Proportion weighting accuracy: %.2f (last), %.2f (average), %.2f" " (best)\n" % ( accuracy["proportion"][-1], np.mean(accuracy["proportion"]), np.max(accuracy["proportion"]), ) )

        print("Progress:",step*batch_size,"/",n_train)

        # Assign labels to excitatory layer neurons.
        assignments, proportions, rates = assign_labels( spikes=spike_record, labels=label_tensor, n_labels=n_classes, rates=rates,
        )

        labels = []

    labels.extend(batch["label"].tolist())

    # Run the network on the input.
    network.run(inputs=inputs, time=time, input_time_dim=1)

    # Add to spikes recording.
    s = output_spikes_monitor.get("s").permute((1, 0, 2))
    spike_record[(step * batch_size) % update_interval : (step * batch_size % update_interval) + s.size(0)] = s

    network.reset_state_variables()  # Reset state variables.

print("Training complete.\n")

# save network
filename = "./networks/snn_" + str(args.encoding) + "_" + str(args.neuron_model) + "_" + str(args.update_rule) + ".pt"
network.save(filename)

# Load MNIST data.
test_dataset = MNIST(
        encoder,
        None,
        root=os.path.join(".", "data", "MNIST"),
        download=True,
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
        ),
    )

# Create a dataloader to iterate and batch data
test_dataloader = DataLoader( test_dataset, batch_size=256, shuffle=False, num_workers=n_workers, pin_memory=True, )

# Sequence of accuracy estimates.
accuracy = {"all": 0, "proportion": 0}

# Train the network.
print("\nBegin testing\n")
network.train(mode=False)

for step, batch in enumerate(test_dataloader):

    # Get next input sample.
    inputs = {"X": batch["encoded_image"].cuda()}
    #inputs = {k: v.cuda() for k, v in inputs.items()}

    # Run the network on the input.
    network.run(inputs=inputs, time=time, input_time_dim=1)

    # Add to spikes recording.
    spike_record = output_spikes_monitor.get("s").permute((1, 0, 2))

    # Convert the array of labels into a tensor
    label_tensor = torch.tensor(batch["label"], device=device)

    # Get network predictions.
    all_activity_pred = all_activity( spikes=spike_record, assignments=assignments, n_labels=n_classes )
    proportion_pred = proportion_weighting( spikes=spike_record, assignments=assignments, proportions=proportions, n_labels=n_classes, )

    # Compute network accuracy according to available classification strategies.
    accuracy["all"] += float(torch.sum(label_tensor.long() == all_activity_pred).item())
    accuracy["proportion"] += float( torch.sum(label_tensor.long() == proportion_pred).item() )

    network.reset_state_variables()  # Reset state variables.
    if step % update_steps == 0 and step > 0:
        print("\nAll activity accuracy: %.2f" % (accuracy["all"] / n_test))
        print("Proportion weighting accuracy: %.2f \n" % (accuracy["proportion"] / n_test))
        print("Progress:",step*256,"/",n_train)

print("\nAll activity accuracy: %.2f" % (accuracy["all"] / n_test))
print("Proportion weighting accuracy: %.2f \n" % (accuracy["proportion"] / n_test))

print("Testing complete.\n")
