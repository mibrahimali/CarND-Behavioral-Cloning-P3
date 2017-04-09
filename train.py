import data_generator
import models
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
# from keras.utils.visualize_util import plot
import matplotlib.pyplot as plt

# model = models.lenet_model((160, 320, 3))
model = models.nvidia_model((160, 320, 3))

modelcheckpoint_cb = ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', save_best_only=True, period=1)


optimizer = Adam(lr=1e-4)
model.compile(optimizer, loss="mse")
# plot(lenet_model, to_file='model.png')
model.summary()

train_log_path = r'/home/mohamed/Udacity/CarND-Term1-Starter-Kit/Projects/CarND-Behavioral-Cloning-P3/Udacity_data/train_log.csv'
valid_log_path = r'/home/mohamed/Udacity/CarND-Term1-Starter-Kit/Projects/CarND-Behavioral-Cloning-P3/Udacity_data/valid_log.csv'


train_generator = data_generator.dataset_generator(batch_size=128, paths=[train_log_path])
valid_generator = data_generator.dataset_generator(batch_size=64, paths=[valid_log_path])

history_object = model.fit_generator(
                                        train_generator,
                                        samples_per_epoch=6400,
                                        nb_epoch=10,
                                        validation_data=valid_generator,
                                        nb_val_samples=4800, callbacks=[modelcheckpoint_cb]

                                    )

model.save("model.h5")

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

