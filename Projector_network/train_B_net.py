import tensorflow as tf
import os
import Net_Arch
import numpy as np
import time
from Net_Arch import slmLength, ccdLength
from tensorflow.keras import backend as K

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# Data path
data_dir = './dataset_04_26/'
activeLength = slmLength // 2
# Learning strategy
learning_rate = 1.0e-4
decay = 1.0e-2
batch_size = 20
num_epochs = 1
buffer_size = 15000
test_num = 1000
MSE = tf.keras.losses.MeanSquaredError()
pi = np.pi
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)


def get_color_matrix(shape):
    width = shape[1]
    color_matrix = np.ones(shape)
    for i in range(shape[0]):
        for color in range(3):
            color_matrix[i, 0:width // 3 + 2, :, color] = color_matrix[i, 0:width // 3 + 2, :,
                                                          color] * np.random.uniform()
            color_matrix[i, width // 3 + 2:width // 3 * 2, :, color] = color_matrix[i, width // 3 + 2:width // 3 * 2, :,
                                                                       color] * np.random.uniform()
            color_matrix[i, width // 3 * 2:, :, color] = color_matrix[i, width // 3 * 2:, :,
                                                         color] * np.random.uniform()
    return color_matrix


def B_net_Loss(y_true, y_pred):
    speckle_pred = F_net_RGB(y_pred)
    speckle_pred = tf.slice(speckle_pred,
                            begin=[0, (ccdLength - activeLength) // 2, (ccdLength - activeLength) // 2, 0],
                            size=[-1, activeLength, activeLength, -1])
    mse = MSE(y_true, speckle_pred)
    return mse


class B_net_PCC(tf.keras.metrics.Metric):
    def __init__(self):
        super().__init__()
        self.pearsonCorr = self.add_weight(name='pearsonCorr', dtype=tf.float32, initializer=tf.zeros_initializer())

    def update_state(self, y_true, y_pred, sample_weight=None):
        speckle_pred = F_net_RGB(y_pred)
        speckle_pred = tf.slice(speckle_pred,
                                begin=[0, (ccdLength - activeLength) // 2, (ccdLength - activeLength) // 2, 0],
                                size=[-1, activeLength, activeLength, -1])
        x = tf.reshape(y_true, [-1])
        y = tf.reshape(speckle_pred, [-1])
        mx = K.mean(x)
        my = K.mean(y)
        xm, ym = x - mx, y - my
        r_num = K.sum(tf.multiply(xm, ym))
        r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
        r = r_num / r_den
        self.pearsonCorr.assign(r)

    def result(self):
        return self.pearsonCorr


def train_B_net(F_net_RGB, F_net_save_dir, B_net, B_net_save_dir, train_dataset, test_dataset, log_dir):
    F_net_RGB.compile(
        optimizer=optimizer,
        loss=MSE,
        run_eagerly=True,
    )
    F_net_RGB.build(input_shape=[batch_size, slmLength, slmLength, 1])
    F_net_RGB.call(tf.keras.Input([slmLength, slmLength, 1]))
    F_net_RGB.summary()
    F_net_RGB.F_net_R.load_weights(F_net_save_dir + 'red/F_net_red.keras', by_name=False)
    F_net_RGB.F_net_G.load_weights(F_net_save_dir + 'green/F_net_green.keras', by_name=False)
    F_net_RGB.F_net_B.load_weights(F_net_save_dir + 'blue/F_net_blue.keras', by_name=False)

    B_net.compile(
        optimizer=optimizer,
        loss=B_net_Loss,
        metrics=[PCC],
        run_eagerly=True,
    )
    B_net.build(input_shape=[batch_size, activeLength, activeLength, 3])
    B_net.call(tf.keras.Input([activeLength, activeLength, 3]))
    B_net.summary()
    history = B_net.fit(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        epochs=num_epochs,
        validation_data=test_dataset,
        validation_freq=1,
    )
    B_net.save_weights(B_net_save_dir)

    for key in history.history.keys():
        with open(log_dir + key + '.txt', 'w') as f:
            f.write(key + ':\n')
            for value in history.history[key]:
                f.write(str(value) + '\n')


if __name__ == '__main__':

    mnist_train = np.load(data_dir + 'train_B_net/mnist/train_data.npy')
    np.random.shuffle(mnist_train)
    mnist_train = mnist_train[0:buffer_size]
    mnist_train = np.expand_dims(mnist_train, axis=-1).repeat(3, axis=-1)
    mnist_train = tf.image.resize(mnist_train, [activeLength, activeLength])
    color_matrix = get_color_matrix(mnist_train.shape)
    mnist_train = tf.multiply(mnist_train, color_matrix)

    mnist_test = np.load(data_dir + 'train_B_net/mnist/test_data.npy')
    mnist_test = mnist_test[0:test_num, :]
    mnist_test = np.expand_dims(mnist_test, axis=-1).repeat(3, axis=-1)
    mnist_test = tf.image.resize(mnist_test, [activeLength, activeLength])
    color_matrix = get_color_matrix(mnist_test.shape)
    mnist_test = tf.multiply(mnist_test, color_matrix)

    # Train
    train_dataset = tf.data.Dataset.from_tensor_slices((mnist_train, mnist_train))
    train_dataset = train_dataset.shuffle(buffer_size=buffer_size)
    train_dataset = train_dataset.batch(batch_size=batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((mnist_test, mnist_test))
    test_dataset = test_dataset.shuffle(buffer_size=buffer_size)
    test_dataset = test_dataset.batch(batch_size=batch_size)

    F_net_RGB = Net_Arch.F_net_RGB()
    B_net = Net_Arch.B_net()
    PCC = B_net_PCC()
    train_B_net(F_net_RGB=F_net_RGB,
                F_net_save_dir=data_dir + 'train_F_net/F_net_log/',
                B_net=B_net,
                B_net_save_dir=data_dir + 'train_B_net/B_net_log/B_net.keras',
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                log_dir=data_dir + 'train_B_net/B_net_log/')
