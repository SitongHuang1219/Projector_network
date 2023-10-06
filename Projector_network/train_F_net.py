import tensorflow as tf
import os
import Net_Arch
from Net_Arch import slmLength, ccdLength
import numpy as np

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# Data path
data_dir = './dataset_04_26/train_F_net/'
# Learning strategy
learning_rate = 1.0e-4
decay = 1.0e-2
batch_size = 20
num_epochs = 1
buffer_size = 15000
test_num = 1000
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss_func = tf.keras.losses.MeanSquaredError()
PCC = Net_Arch.PCC()

pi = np.pi


def F_net_map_func(image, speckle):
    speckle = tf.reshape(speckle, [ccdLength, ccdLength, 1])
    image = tf.reshape(image, [slmLength, slmLength, 1])
    return image, speckle


def train_F_net(color, save_dir, train_dataset, test_dataset, log_dir):
    F_net = Net_Arch.F_net(name=color)
    F_net.compile(
        optimizer=optimizer,
        loss=loss_func,
        metrics=[PCC],
        run_eagerly=True,
    )
    F_net.build(input_shape=[batch_size, slmLength, slmLength, 1])
    F_net.call(tf.keras.Input([slmLength, slmLength, 1]))
    F_net.summary()

    history = F_net.fit(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        epochs=num_epochs,
        validation_data=test_dataset,
        validation_freq=1,
    )
    F_net.save_weights(save_dir)

    for key in history.history.keys():
        with open(log_dir + key + '.txt', 'w') as f:
            f.write(key + ':\n')
            for value in history.history[key]:
                f.write(str(value) + '\n')


if __name__ == '__main__':

    # Load dataset
    speckles = np.load(data_dir + '/speckle_data.npy')
    speckles = np.split(speckles, 3, axis=-1)

    phase_maps = np.load(data_dir + '/phase_map_data.npy')

    # Train
    color_list = ['red', 'green', 'blue']
    for i in range(len(color_list)):
        color = color_list[i]
        train_dataset = tf.data.Dataset.from_tensor_slices((phase_maps[0:buffer_size], speckles[i][0:buffer_size]))
        train_dataset = train_dataset.map(map_func=F_net_map_func)
        train_dataset = train_dataset.shuffle(buffer_size=buffer_size)
        train_dataset = train_dataset.batch(batch_size=batch_size)

        test_dataset = tf.data.Dataset.from_tensor_slices(
            (phase_maps[buffer_size:buffer_size + test_num], speckles[i][buffer_size:buffer_size + test_num]))
        test_dataset = test_dataset.map(map_func=F_net_map_func)
        test_dataset = test_dataset.shuffle(buffer_size=buffer_size)
        test_dataset = test_dataset.batch(batch_size=batch_size)

        train_F_net(color=color,
                save_dir=data_dir + 'F_net_log/' + color + '/F_net_' + color + '.keras',
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                log_dir=data_dir + 'F_net_log/' + color + '/')
