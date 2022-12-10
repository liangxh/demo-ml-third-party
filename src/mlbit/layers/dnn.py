# 参考 from deepctr.layers.core import DNN

from tensorflow.python.ops.init_ops_v2 import Zeros, glorot_normal
from tensorflow.python.keras.layers import Layer, Dropout, BatchNormalization
from tensorflow.python.keras.regularizers import l2

from mlbit.layers.activation import ActivationFactory


class DNN(Layer):
    def __init__(
            self,
            hidden_units,
            activation='relu',
            l2_reg=0,
            dropout_rate=0,
            use_bn=False,
            output_activation=None,
            seed=1024,
            **kwargs):
        super(DNN, self).__init__(**kwargs)
        self.hidden_units = hidden_units
        self.activation = activation
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.output_activation = output_activation
        self.seed = seed

        self.kernels = None
        self.bias = None
        self.bn_layers = None
        self.dropout_layers = None
        self.activation_layers = None

    def build(self, input_shape):
        # if len(self.hidden_units) == 0:
        #     raise ValueError("hidden_units is empty")

        input_size = input_shape[-1]
        hidden_units = [int(input_size)] + list(self.hidden_units)
        self.kernels = [
            self.add_weight(
                name='kernel' + str(i),
                shape=(hidden_units[i], hidden_units[i + 1]),
                initializer=glorot_normal(seed=self.seed),
                regularizer=l2(self.l2_reg),
                trainable=True
            )
            for i in range(len(self.hidden_units))
        ]
        self.bias = [
            self.add_weight(
                name='bias' + str(i),
                shape=(self.hidden_units[i],),
                initializer=Zeros(),
                trainable=True
            )
            for i in range(len(self.hidden_units))
        ]
        if self.use_bn:
            self.bn_layers = [BatchNormalization() for _ in range(len(self.hidden_units))]

        self.dropout_layers = [
            Dropout(self.dropout_rate, seed=self.seed + i)
            for i in range(len(self.hidden_units))
        ]

        self.activation_layers = [
            ActivationFactory.build(self.activation)
            for _ in range(len(self.hidden_units))]

        if self.output_activation:
            self.activation_layers[-1] = ActivationFactory.build(self.output_activation)

        super(DNN, self).build(input_shape)
