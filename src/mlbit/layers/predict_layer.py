# deepctr.layers.core.PredictionLayer

import tensorflow as tf
from tensorflow.python.ops.init_ops_v2 import Zeros
from tensorflow.python.keras.layers import Layer


class PredictionLayer(Layer):
    """
      Arguments
         - **task**: str,
            ``"binary"`` for  binary logloss
            ``"regression"`` for regression loss
         - **use_bias**: bool.
         Whether add bias term or not.
    """

    def __init__(self, task='binary', use_bias=True, **kwargs):
        if task not in ["binary", "multiclass", "regression"]:
            raise ValueError("task must be binary,multiclass or regression")
        self.task = task
        self.use_bias = use_bias
        self.global_bias = None  # Layer
        super(PredictionLayer, self).__init__(**kwargs)

    # # override
    def build(self, input_shape):
        if self.use_bias:
            self.global_bias = self.add_weight(
                shape=(1,),
                initializer=Zeros(),
                name="global_bias"
            )
        super(PredictionLayer, self).build(input_shape)

    # # override
    def call(self, inputs, **kwargs):
        x = inputs
        if self.use_bias:
            x = tf.nn.bias_add(x, self.global_bias, data_format='NHWC')
        if self.task == "binary":
            x = tf.sigmoid(x)

        # 最后输出 N 行, 每行一维
        output = tf.reshape(x, (-1, 1))

        return output

    # override, 输出矩阵大小, 每一行只有一维
    def compute_output_shape(self, input_shape):
        return (None, 1)

    # override, 返回模型配置
    def get_config(self, ):
        config = {'task': self.task, 'use_bias': self.use_bias}
        base_config = super(PredictionLayer, self).get_config()
        base_config.update(config)
        return base_config
