from typing import Optional, Iterable

import torch
from bindsnet.learning import LearningRule, PostPre
from bindsnet.network import Network
from bindsnet.network.nodes import Input, IFNodes, LIFNodes, Nodes, SRM0Nodes, DiehlAndCookNodes
from bindsnet.network.topology import Connection

# RecurrentNetwork has the structure of the Diehl and Cook (2015) network, but allows a user to specifiy the neural model and learning rule
class RecurrentNetwork(Network):

    # RecurrentNetwork's constructor takes in the number of inputs, the output layer Nodes object, the update rule object, the batch size, and the input shape
    def __init__(
        self,
        n_inputs: int,
        output_layer: Nodes,
        update_rule: Optional[LearningRule] = PostPre,
        batch_size: int = 1,
        input_shape: Optional[Iterable[int]] = None,
    ) -> None:

        # the batch size is passed to the super class Network
        super().__init__(batch_size=batch_size)

        # the number of output neurons in this network is always 100
        n_neurons = 100

        # create the input layer with the specified number of inputs and input shape
        # specify the spike traces are enabled and set the time constant of the spike trace to 20
        input_layer = Input( n=n_inputs, shape=input_shape, traces=True, tc_trace=20.0 )

        # add the input layer to the network with the name X
        self.add_layer(input_layer, name="X")

        # add the output layer to the network with the name Y
        self.add_layer(output_layer, name="Y")

        # initialize the weights of the synapses connecting the input layer to the output layer
        w = 0.3 * torch.rand(n_inputs, n_neurons)

        # create a new connection between the input layer and the output layer
        # specify the update rule, learning rate (nu), minimum/maximum weight values, and the normalization constant
        input_connection = Connection(
            source=self.layers["X"],
            target=self.layers["Y"],
            w=w,
            update_rule=update_rule,
            nu=(1e-4, 1e-2),
            wmin=0,
            wmax=1,
            norm=78.4,
        )
        # add the input-output layer connection to the network
        self.add_connection(input_connection, source="X", target="Y")

        # initialize the weights of the recurrent inhibitory synapses connecting the output layer to itself
        # this tensor is organized in such a way that a neuron does not have an inhibitory connection to itself
        w = -120 * (
            torch.ones(n_neurons, n_neurons)
            - torch.diag(torch.ones(n_neurons))
        )

        # create the inhibitory connections at the output layer with a maximum weight of 0 and a minimum weight of -120
        recurrent_connection = Connection(
            source=self.layers["Y"],
            target=self.layers["Y"],
            w=w,
            wmin=-120,
            wmax=0,
        )

        # add the inhibitory output layer connection to the network
        self.add_connection(recurrent_connection, source="Y", target="Y")

# IF_Network is subclass of the previously established RecurrentNetwork that uses IF neurons at the output layer
class IF_Network(RecurrentNetwork):
    def __init__(
        self,
        n_inputs: int,
        input_shape: Optional[Iterable[int]] = None,
        update_rule: Optional[LearningRule] = PostPre,
        batch_size: int = 1,
    ) -> None:

        # create a new output layer of 100 IF Nodes with spike traces enabled and a reset value of -60
        output_layer = IFNodes(
            n=100,
            traces=True,
            reset=-60.0,
        )

        # pass the arguments to the RecurrentNetwork constructor
        super().__init__(n_inputs=n_inputs,output_layer=output_layer,input_shape=input_shape,update_rule=update_rule,batch_size=batch_size)

# LIF_Network is subclass of the previously established RecurrentNetwork that uses LIF neurons at the output layer
class LIF_Network(RecurrentNetwork):
    def __init__(
        self,
        n_inputs: int,
        input_shape: Optional[Iterable[int]] = None,
        update_rule: Optional[LearningRule] = PostPre,
        batch_size: int = 1,
    ) -> None:


        # create a new output layer of 100 LIF Nodes with spike traces enabled and a reset value of -60
        output_layer = LIFNodes(
            n=100,
            traces=True,
            reset=-60.0,
        )

        # pass the arguments to the RecurrentNetwork constructor
        super().__init__(n_inputs=n_inputs,output_layer=output_layer,input_shape=input_shape,update_rule=update_rule,batch_size=batch_size)

# SRM0_Network is subclass of the previously established RecurrentNetwork that uses SRM0 neurons at the output layer
class SRM0_Network(RecurrentNetwork):
    def __init__(
        self,
        n_inputs: int,
        input_shape: Optional[Iterable[int]] = None,
        update_rule: Optional[LearningRule] = PostPre,
        batch_size: int = 1,
    ) -> None:
        
        # create a new output layer of 100 SRM0 Nodes with spike traces enabled, a reset value of -60, a resting voltage of -65, and a threshold of -52
        output_layer = SRM0Nodes(
            n=100,
            traces=True,
            rest=-65.0,
            reset=-60.0,
            thresh=-52.0,
        )

        # pass the arguments to the RecurrentNetwork constructor
        super().__init__(n_inputs=n_inputs,output_layer=output_layer,input_shape=input_shape,update_rule=update_rule,batch_size=batch_size)

# DiehlAndCook_Network is subclass of the previously established RecurrentNetwork that uses SRM0 neurons at the output layer
class DiehlAndCook_Network(RecurrentNetwork):
    def __init__(
        self,
        n_inputs: int,
        input_shape: Optional[Iterable[int]] = None,
        update_rule: Optional[LearningRule] = PostPre,
        batch_size: int = 1,
    ) -> None:

        # create a new output layer of 100 DiehlAndCook Nodes with spike traces enabled and a reset value of -60
        output_layer = DiehlAndCookNodes(
            n=100,
            traces=True,
            reset=-60.0,
        )

        # pass the arguments to the RecurrentNetwork constructor
        super().__init__(n_inputs=n_inputs,output_layer=output_layer,input_shape=input_shape,update_rule=update_rule,batch_size=batch_size)