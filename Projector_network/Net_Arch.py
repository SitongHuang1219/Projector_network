import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K

pi = np.pi
B_net_units = 2048
F_net_units = 1024
slmLength = 40
ccdLength = 80
initialize_maxval = np.sqrt(1 / F_net_units / 2)


class F_net(tf.keras.Model):
    def __init__(self, name='name'):
        super().__init__()
        self.Flatten = tf.keras.layers.Flatten()
        self.Cosine = tf.keras.layers.Lambda(function=K.cos)
        self.Sine = tf.keras.layers.Lambda(function=K.sin)
        self.DenseTr1 = tf.keras.layers.Dense(units=F_net_units,
                                              activation=tf.nn.tanh,
                                              kernel_initializer=tf.keras.initializers.RandomUniform(
                                                  minval=-initialize_maxval,
                                                  maxval=initialize_maxval),
                                              name='DenseTr1_' + name,
                                              )
        self.DenseTr2 = tf.keras.layers.Dense(units=ccdLength * ccdLength,
                                              kernel_initializer=tf.keras.initializers.RandomUniform(
                                                  minval=-initialize_maxval,
                                                  maxval=initialize_maxval),
                                              name='DenseTr2_' + name,
                                              )
        self.DenseTi1 = tf.keras.layers.Dense(units=F_net_units,
                                              activation=tf.nn.tanh,
                                              kernel_initializer=tf.keras.initializers.RandomUniform(
                                                  minval=-initialize_maxval,
                                                  maxval=initialize_maxval),
                                              name='DenseTi1_' + name,
                                              )
        self.DenseTi2 = tf.keras.layers.Dense(units=ccdLength * ccdLength,
                                              kernel_initializer=tf.keras.initializers.RandomUniform(
                                                  minval=-initialize_maxval,
                                                  maxval=initialize_maxval),
                                              name='DenseTi2_' + name,
                                              )

        self.R_Add = tf.keras.layers.Add()
        self.I_Add = tf.keras.layers.Add()
        self.R_Square = tf.keras.layers.Lambda(function=tf.square)
        self.I_Square = tf.keras.layers.Lambda(function=tf.square)

        self.R_I_Add = tf.keras.layers.Add()
        self.Reshape = tf.keras.layers.Reshape(target_shape=[ccdLength, ccdLength, 1])

    def call(self, inputs, training=None, mask=None):
        x = self.Flatten(inputs)
        Ir = self.Cosine(2. * pi * x)
        Ii = self.Sine(2. * pi * x)

        R1 = self.DenseTr1(Ir)
        R1 = self.DenseTr2(R1)
        R2 = self.DenseTi1(Ii)
        R2 = self.DenseTi2(R2)
        R = self.R_Add([R1, -1. * R2])
        R = self.R_Square(R)

        I1 = self.DenseTr1(Ii)
        I1 = self.DenseTr2(I1)
        I2 = self.DenseTi1(Ir)
        I2 = self.DenseTi2(I2)
        I = self.I_Add([I1, I2])
        I = self.I_Square(I)

        outputs = self.R_I_Add([R, I])
        outputs = self.Reshape(outputs)

        return outputs


class F_net_RGB(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.F_net_R = F_net(name='red')
        self.F_net_G = F_net(name='green')
        self.F_net_B = F_net(name='blue')
        self.Concat = tf.keras.layers.Concatenate(axis=-1)

    def call(self, inputs, training=None, mask=None):
        outputs_R = self.F_net_R(inputs, training=False)
        outputs_G = self.F_net_G(inputs, training=False)
        outputs_B = self.F_net_B(inputs, training=False)

        outputs = self.Concat([outputs_R,
                               outputs_G,
                               outputs_B])

        return outputs


class CobComplex(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CobComplex, self).__init__()
        # self.pred = kwargs.get('pred', False)

    def call(self, Real, Imag):
        return tf.complex(Real, Imag)


class B_net(tf.keras.Model):
    def __init__(self, name='name'):
        super().__init__()
        self.Flatten = tf.keras.layers.Flatten()
        self.Sqrt = tf.keras.layers.Lambda(function=tf.sqrt)
        self.RealDense_1 = tf.keras.layers.Dense(units=B_net_units,
                                                 activation=tf.nn.leaky_relu,
                                                 kernel_initializer=tf.keras.initializers.RandomUniform(
                                                     minval=-initialize_maxval,
                                                     maxval=initialize_maxval),
                                                 name='RealDense_1_' + name,
                                                 )
        self.RealDense_2 = tf.keras.layers.Dense(units=B_net_units,
                                                 activation=tf.nn.tanh,
                                                 kernel_initializer=tf.keras.initializers.RandomUniform(
                                                     minval=-initialize_maxval,
                                                     maxval=initialize_maxval),
                                                 name='RealDense_2_' + name,
                                                 )
        self.RealDense_3_a = tf.keras.layers.Dense(units=slmLength * slmLength,
                                                   kernel_initializer=tf.keras.initializers.RandomUniform(
                                                       minval=-initialize_maxval,
                                                       maxval=initialize_maxval),
                                                   name='RealDense_3_a_' + name,
                                                   )
        self.RealDense_3_b = tf.keras.layers.Dense(units=slmLength * slmLength,
                                                   kernel_initializer=tf.keras.initializers.RandomUniform(
                                                       minval=-initialize_maxval,
                                                       maxval=initialize_maxval),
                                                   name='RealDense_3_b_' + name,
                                                   )

        self.ImagDense_1 = tf.keras.layers.Dense(units=B_net_units,
                                                 activation=tf.nn.leaky_relu,
                                                 kernel_initializer=tf.keras.initializers.RandomUniform(
                                                     minval=-initialize_maxval,
                                                     maxval=initialize_maxval),
                                                 name='ImagDense_1_' + name,
                                                 )
        self.ImagDense_2 = tf.keras.layers.Dense(units=B_net_units,
                                                 activation=tf.nn.tanh,
                                                 kernel_initializer=tf.keras.initializers.RandomUniform(
                                                     minval=-initialize_maxval,
                                                     maxval=initialize_maxval),
                                                 name='ImagDense_2_' + name,
                                                 )
        self.ImagDense_3_a = tf.keras.layers.Dense(units=slmLength * slmLength,
                                                   kernel_initializer=tf.keras.initializers.RandomUniform(
                                                       minval=-initialize_maxval,
                                                       maxval=initialize_maxval),
                                                   name='ImagDense_3_a_' + name,
                                                   )
        self.ImagDense_3_b = tf.keras.layers.Dense(units=slmLength * slmLength,
                                                   kernel_initializer=tf.keras.initializers.RandomUniform(
                                                       minval=-initialize_maxval,
                                                       maxval=initialize_maxval),
                                                   name='ImagDense_3_b_' + name,
                                                   )
        self.Add_a = tf.keras.layers.Add()
        self.Act_a = tf.keras.layers.Activation(tf.nn.tanh)
        self.Add_b = tf.keras.layers.Add()
        self.Act_b = tf.keras.layers.Activation(tf.nn.tanh)
        self.Complex = CobComplex()
        self.Angle = tf.keras.layers.Lambda(function=tf.math.angle)
        self.Shift = tf.keras.layers.Lambda(function=(lambda x: (x + pi) / 2 / pi))
        self.Reshape = tf.keras.layers.Reshape(target_shape=[slmLength, slmLength, 1])

    def call(self, inputs):
        x = self.Flatten(inputs)
        x = self.Sqrt(x)

        r = self.RealDense_1(x)
        r = self.RealDense_2(r)
        R_a = self.RealDense_3_a(r)
        R_b = self.RealDense_3_b(r)

        i = self.ImagDense_1(x)
        i = self.ImagDense_2(i)
        I_a = self.ImagDense_3_a(i)
        I_b = self.ImagDense_3_b(i)

        R = self.Add_a([R_a, I_a])
        R = self.Act_a(R)
        I = self.Add_b([R_b, I_b])
        I = self.Act_b(I)
        x = self.Complex(R, I)
        x = self.Angle(x)
        x = self.Shift(x)
        outputs = self.Reshape(x)
        return outputs


class PCC(tf.keras.metrics.Metric):
    def __init__(self):
        super().__init__()
        self.pearsonCorr = self.add_weight(name='pearsonCorr', dtype=tf.float32, initializer=tf.zeros_initializer())

    def update_state(self, y_true, y_pred, sample_weight=None):
        x = tf.reshape(y_true, [-1])
        y = tf.reshape(y_pred, [-1])
        mx = K.mean(x)
        my = K.mean(y)
        xm, ym = x - mx, y - my
        r_num = K.sum(tf.multiply(xm, ym))
        r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
        r = r_num / r_den
        self.pearsonCorr.assign(r)

    def result(self):
        return self.pearsonCorr


if __name__ == '__main__':
    f_net = F_net(name='red')
    loss_func = tf.keras.losses.MeanSquaredError()
    f_net.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1),
        loss=loss_func,
        metrics=[]
    )
    f_net.build(input_shape=[1, slmLength, slmLength, 1])
    f_net.summary()

    f_net_RGB = F_net_RGB()
    loss_func = tf.keras.losses.MeanSquaredError()
    f_net_RGB.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1),
        loss=loss_func,
        metrics=[]
    )
    f_net_RGB.build(input_shape=[1, slmLength, slmLength, 1])
    f_net_RGB.summary()

    b_net = B_net()
    loss_func = tf.keras.losses.MeanSquaredError()
    b_net.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1),
        loss=loss_func,
        metrics=[]
    )
    b_net.build(input_shape=[500, ccdLength, ccdLength, 3])
    b_net.summary()
