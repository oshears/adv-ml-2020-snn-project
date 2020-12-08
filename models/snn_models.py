from typing import Optional, Union, Sequence, Iterable

import torch
from bindsnet.learning import LearningRule, PostPre
from bindsnet.network import Network
from bindsnet.network.nodes import Input, IFNodes, LIFNodes, Nodes, SRM0Nodes, DiehlAndCookNodes
from bindsnet.network.topology import Connection

class RecurrentNetwork(Network):
    def __init__(
        self,
        n_inputs: int,
        output_layer: Nodes,
        update_rule: Optional[LearningRule] = PostPre,
        batch_size: int = 1,
        input_shape: Optional[Iterable[int]] = None,
    ) -> None:
        super().__init__(batch_size=batch_size)

        n_neurons = 100

        input_layer = Input(
            n=n_inputs, shape=input_shape, traces=True, tc_trace=20.0
        )
        self.add_layer(input_layer, name="X")

        self.add_layer(output_layer, name="Y")

        w = 0.3 * torch.rand(n_inputs, n_neurons)
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
        self.add_connection(input_connection, source="X", target="Y")

        w = -120 * (
            torch.ones(n_neurons, n_neurons)
            - torch.diag(torch.ones(n_neurons))
        )
        recurrent_connection = Connection(
            source=self.layers["Y"],
            target=self.layers["Y"],
            w=w,
            wmin=-120,
            wmax=0,
        )
        self.add_connection(recurrent_connection, source="Y", target="Y")


class IF_Network(RecurrentNetwork):
    def __init__(
        self,
        n_inputs: int,
        dt: float = 1.0,
        input_shape: Optional[Iterable[int]] = None,
        update_rule: Optional[LearningRule] = PostPre,
        batch_size: int = 1,
    ) -> None:

        output_layer = IFNodes(
            n=100,
            traces=True,
            reset=-60.0,
        )

        super().__init__(n_inputs=n_inputs,output_layer=output_layer,input_shape=input_shape,update_rule=update_rule,batch_size=batch_size)

class LIF_Network(RecurrentNetwork):
    def __init__(
        self,
        n_inputs: int,
        dt: float = 1.0,
        input_shape: Optional[Iterable[int]] = None,
        update_rule: Optional[LearningRule] = PostPre,
        batch_size: int = 1,
    ) -> None:


        n_neurons = 100

        output_layer = LIFNodes(
            n=n_neurons,
            traces=True,
            reset=-60.0,
        )

        super().__init__(n_inputs=n_inputs,output_layer=output_layer,input_shape=input_shape,update_rule=update_rule,batch_size=batch_size)

class SRM0_Network(RecurrentNetwork):
    def __init__(
        self,
        n_inputs: int,
        input_shape: Optional[Iterable[int]] = None,
        update_rule: Optional[LearningRule] = PostPre,
        batch_size: int = 1,
    ) -> None:
        
        output_layer = SRM0Nodes(
            n=100,
            traces=True,
            rest=-65.0,
            reset=-60.0,
            thresh=-52.0,
        )

        super().__init__(n_inputs=n_inputs,output_layer=output_layer,input_shape=input_shape,update_rule=update_rule,batch_size=batch_size)

class DiehlAndCook_Network(Network):
    def __init__(
        self,
        n_inputs: int,
        input_shape: Optional[Iterable[int]] = None,
        update_rule: Optional[LearningRule] = PostPre,
        batch_size: int = 1,
    ) -> None:

        output_layer = DiehlAndCookNodes(
            n=100,
            traces=True,
            reset=-60.0,
        )

        super().__init__(n_inputs=n_inputs,output_layer=output_layer,input_shape=input_shape,update_rule=update_rule,batch_size=batch_size)