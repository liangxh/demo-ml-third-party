# 参考 from deepctr.layers.activation import Dice

import tensorflow as tf
from tensorflow.python.ops.init_ops import Zeros
from tensorflow.python.keras.layers import Layer, BatchNormalization


class Dice(Layer):
    def __init__(self, axis=-1, epsilon=1e-9, **kwargs):
        super(Dice, self).__init__(**kwargs)

        self.axis = axis
        self.epsilon = epsilon

        self.bn = None      # Layer
        self.alphas = None  # Layer
        self.uses_learning_phase = True

    def build(self, input_shape):
        super(Dice, self).build(input_shape)

        self.bn = BatchNormalization(
            axis=self.axis,
            epsilon=self.epsilon,
            center=False,
            scale=False
        )

        self.alphas = self.add_weight(
            shape=(input_shape[-1], ),   # N 列则 N 个参数
            initializer=Zeros(),         # 以零初始化
            dtype=tf.float32,
            name='dice_alpha'
        )

    def call(self, inputs, training=None, **kwargs):
        inputs_normed = self.bn(inputs, training=training)
        # x_p 在原文中定义为 I(x > 0)
        # 此处当 x > 0, x_p -> 1
        #     当 x < 0, x_p -> 0
        x_p = tf.sigmoid(inputs_normed)
        return self.alphas * (1.0 - x_p) * inputs + x_p * inputs

    # override, 输出矩阵大小, 输出行数列数和输入一致
    def compute_output_shape(self, input_shape):
        return input_shape

    # override, 返回模型配置
    def get_config(self):
        config = super(Dice, self).get_config()
        config.update({'axis': self.axis, 'epsilon': self.epsilon})
        return config
