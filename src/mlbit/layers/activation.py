
from tensorflow.python.keras.layers import Layer, Activation
from mlbit.layers.dice import Dice


class ActivationFactory:
    @classmethod
    def build(cls, activation):
        if activation in ("dice", "Dice"):
            act_layer = Dice()
        elif isinstance(activation, str):
            act_layer = Activation(activation)
        elif issubclass(activation, Layer):
            act_layer = activation()
        else:
            raise ValueError("Invalid activation,found %s.You should use a str or a Activation Layer Class." % (activation))
        return act_layer
