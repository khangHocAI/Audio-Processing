import tensorflow as tf
import numpy as np
from logmelspec import LogMelSpectrogram
import librosa
from tensorflow.keras import metrics
import pylab
import librosa.display
inputs = tf.keras.Input((4800,))
net = LogMelSpectrogram()(inputs)

model = tf.keras.Model(inputs, net)
loss = tf.keras.losses.MeanSquaredError()
optimizer = tf.optimizers.Adam(learning_rate=0.01,beta_1=0.8, beta_2=0.8, epsilon=1e-4)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[metrics.MeanSquaredError()])

y,sr = librosa.load("1.flac",sr = 16000)
y = y[:4800]
y = np.expand_dims(y, axis=0)
out = model.predict(y)
out = out[0]
print(out)
save_path = 'test.jpg'

pylab.axis('off') # no axis
pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge

librosa.display.specshow(out)
pylab.savefig(save_path, bbox_inches=None, pad_inches=0)
pylab.close()
