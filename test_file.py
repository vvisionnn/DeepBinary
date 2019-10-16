# from tensorflow import keras
# from Deep_binary.unet import unet
#
# model = unet()
# model.summary()
#
# model.compile(
#     optimizer=keras.optimizers.Adam(learning_rate=1e-4),
#     loss='binary_crossentropy',
#     metrics=['accuracy']
# )
#
# history = model.fit(ds, validation_split=0.2, epochs=3, steps_per_epoch=10)

import numpy as np

test = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']

test = np.asarray(test)
ts = np.arange(10)
print(test)
np.random.shuffle(ts)
print(test[ts])



