# 参考 from deepctr.layers.interaction import CrossNet

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.ops.init_ops import Zeros, glorot_normal_initializer

class CrossNet(Layer):
    """
    输入
      2D tensor: (batch_size 批大小, units 特征维数)

    输出
    - 2D tensor: (batch_size 批大小, units 特征维数)

    Arguments
    - layer_name: int, cross net 层数
    - l2_reg: float, [0, 1] 核权重的 L2 正则项权重
    - parameterization: string, "vector" 或 "matrix"
    - seed: int, 随机种子

    References
    - [Wang R, Fu B, Fu G, et al. Deep & cross network for ad click predictions[C]
      Proceedings of the ADKDD'17. ACM, 2017: 12.](https://arxiv.org/abs/1708.05123)
    """

    def __init__(self, layer_num=2, parameterization='vector', l2_reg=0, seed=1024, **kwargs):
        self.layer_num = layer_num
        self.parameterization = parameterization
        self.l2_reg = l2_reg
        self.seed = seed

        # 权重
        self.kernels = None  # [Layer, ]
        self.bias = None     # [Layer, ]
        print('CrossNet parameterization:', self.parameterization)
        super(CrossNet, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise ValueError("要求输入 2 维矩阵，实际输入维数: %d" % (len(input_shape)))

        if not {"vector", "matrix"}.contains(self.parameterization):
            raise ValueError("parameterization 必须是 'vector' 或 'matrix'")

        dim = int(input_shape[-1])
        self.kernels = [
            self.add_weight(
                name='kernel' + str(i),
                shape=(
                    dim,
                    {"vector": 1, "matrix": dim}[self.parameterization]
                ),
                # glorot_normal_initializer: mean = 0, stddev = sqrt(2 / (输入单元个数 + 输出单元个数))
                initializer=glorot_normal_initializer(seed=self.seed),
                regularizer=l2(self.l2_reg),
                trainable=True
            ) for i in range(self.layer_num)
        ]
        self.bias = [
            self.add_weight(
                name='bias' + str(i),
                shape=(dim, 1),
                initializer=Zeros(),  # 偏移量以零初始化
                trainable=True
            ) for i in range(self.layer_num)
        ]
        super(CrossNet, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 2:
            raise ValueError("要求输入 2 维矩阵，实际输入维数: %d" % (K.ndim(inputs)))

        # (batch_size, units) -> (batch_size, units, 1)
        x_0 = tf.expand_dims(inputs, axis=2)
        x_l = x_0
        for i in range(self.layer_num):
            if self.parameterization == 'vector':
                # x_(l+1) = x_0 * (w_l ^T * x_l) ^ T + b_l + x_l
                #         = x_0 * x_l ^T * w_l + b_l + x_l

                # tensordot: x_l (batch_size, units, 1) , kernel (units, 1)
                # x_1 的 units 维度与 kernels 相乘 -> (batch_size, 1, 1)
                xl_w = tf.tensordot(x_l, self.kernels[i], axes=(1, 0))

                # tf.matmul (batch_size, units, 1) , (batch_size, 1, 1)
                dot_ = tf.matmul(x_0, xl_w)

                # (batch_size, units, 1)
                x_l = dot_ + self.bias[i] + x_l
            elif self.parameterization == 'matrix':
                # 核  (units, units)
                # x_l (batch_size, units, 1)
                #
                xl_w = tf.einsum('ij,bjk->bik', self.kernels[i], x_l)  # W * xi  (bs, dim, 1)
                dot_ = xl_w + self.bias[i]                             # W * xi + b
                x_l = x_0 * dot_ + x_l                                 # x0 · (W * xi + b) +xl  Hadamard-product
            else:  # error
                raise ValueError("parameterization should be 'vector' or 'matrix'")

        # (batch_size, units, 1) -> (batch_size, units)
        x_l = tf.squeeze(x_l, axis=2)
        return x_l

    # override, 输出矩阵大小, 每一行只有一维
    def compute_output_shape(self, input_shape):
        return input_shape

    # override, 返回模型配置
    def get_config(self, ):
        config = {
            'layer_num': self.layer_num,
            'parameterization': self.parameterization,
            'l2_reg': self.l2_reg,
            'seed': self.seed
        }
        base_config = super(CrossNet, self).get_config()
        base_config.update(config)
        return base_config

