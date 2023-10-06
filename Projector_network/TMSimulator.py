import tensorflow as tf
import os
import numpy as np
import math
from Net_Arch import slmLength, ccdLength

data_dir = './dataset_04_26/train_F_net/'  # Data path
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
train_num = 15000
test_num = 1000
test_mode = True
Amp = 400.
pi = math.pi

real = tf.random.normal([3, ccdLength * ccdLength, slmLength * slmLength], mean=0, stddev=0.5)
imag = tf.random.normal([3, ccdLength * ccdLength, slmLength * slmLength], mean=0, stddev=0.5)
TM = tf.complex(real, imag) / tf.cast(tf.sqrt(2.), tf.complex64) / tf.cast(slmLength * slmLength, tf.complex64)

phase_map_data = np.zeros([train_num + test_num, slmLength * slmLength, 1])
speckle_data = np.zeros([train_num + test_num, ccdLength * ccdLength, 3])

for i in range(train_num + test_num):
    phase_map = np.random.randint(low=0, high=256, size=[slmLength * slmLength, 1]) / 255.
    phase_map[0, 0] = 96. / 255.
    phase_map_mod = np.repeat(np.expand_dims(phase_map, 0), 3, 0)
    phase_map_mod[0] = phase_map_mod[0] * 1.0
    phase_map_mod[1] = phase_map_mod[1] * 1.2
    phase_map_mod[2] = phase_map_mod[2] * 1.5
    phase_map_mod = np.exp(phase_map_mod)
    speckle = np.matmul(TM, phase_map_mod)
    speckle = Amp * np.square(np.abs(speckle))
    speckle = np.moveaxis(speckle, 0, -1)
    speckle = np.moveaxis(speckle, 1, 0)
    phase_map_data[i] = phase_map
    speckle_data[i] = speckle
    if i % (train_num // 10) == 0:
        print('%.1f%%'.center(30, '-') % (i / train_num * 100))

np.save(data_dir + '/phase_map_data.npy', phase_map_data)
np.save(data_dir + '/speckle_data.npy', speckle_data)
