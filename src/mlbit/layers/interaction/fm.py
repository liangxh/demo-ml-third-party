# 参考 from deepctr.layers.interaction import FM

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer


class FM(Layer):
    """
    输入:
      3D tensor:
        batch_size, 批大小
        field_size, 特征域数
        embedding_size，特征维数 （每个域的特征数一样，以支持交叉）
    输出：2D tensor: (batch_size, 1)

    模型参考论文：Factorization Machines https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf
    """

    def __init__(self, **kwargs):
        super(FM, self).__init__(**kwargs)

    # 准备参数
    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError("要求输入 3 维矩阵，实际输入维数: %d" % (len(input_shape)))
        super(FM, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 3:
            raise ValueError("要求输入 3 维矩阵，实际输入维数: %d" % (K.ndim(inputs)))

        concated_embeds_value = inputs

        # sum((vi * xi) * (vj * xj))
        # = 1/2 * (sum (vi * xi) * sum(vi * xi) - sum((vi * xi)^2))
        # = 1/2 * ((sum(vi * xi))^2 - sum((vi * xi)^2))
        #         --------------          ------------

        # (sum(vi * xi))^2
        square_of_sum = tf.square(      # 各个域的特征按位求和 -> (batch_size, embedding_size)
            # 各维按位相乘      -> (batch_size, 1, embedding_size)
            tf.reduce_sum(concated_embeds_value, axis=1, keepdims=True, name=None)
        )

        sum_of_square = tf.reduce_sum(   # 各维按位平方 -> (batch_size, 1, embedding_size)
            # 各维按位平方      -> (batch_size, field_size, embedding_size)
            concated_embeds_value * concated_embeds_value,
            axis=1, keep_dims=True, name=None
        )

        # -> (batch_size, 1, embedding_size)
        cross_term = square_of_sum - sum_of_square

        # -> (batch_size, 1)
        cross_term = 0.5 * tf.reduce_sum(cross_term, axis=2, keep_dims=False, name=None)

        return cross_term

    def compute_output_shape(self, input_shape):
        return (None, 1)
