import data_generator
import models
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.utils.visualize_util import plot
import matplotlib.pyplot as plt


# Create Instance of Custom Nvidia Model
model = models.nvidia_model((66, 200, 3))

# Create needed callback function
modelcheckpoint_cb = ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', save_best_only=True, period=1)
earlystopping_cb = EarlyStopping(monitor='train_loss', patience=5, verbose=0, mode='auto')


# define training optimizer
adam = Adam(lr=5e-4)
optimizer = adam
# sgd = SGD(lr=1e-4, decay=1e-6)
# optimizer = sgd

# Compiling model with defined optimizer and loss function
model.compile(optimizer, loss="mse")

# save model graph to disk
plot(model, to_file='model.png',show_shapes=True)
model.summary()

# define training and validation dataset paths
train_log_path = r'/home/mohamed/Udacity/CarND-Term1-Starter-Kit/Projects/CarND-Behavioral-Cloning-P3/train_log.csv'
valid_log_path = r'/home/mohamed/Udacity/CarND-Term1-Starter-Kit/Projects/CarND-Behavioral-Cloning-P3/valid_log.csv'

# define datasets generators
train_generator = data_generator.dataset_generator(batch_size=512, paths=[train_log_path])
valid_generator = data_generator.dataset_generator(batch_size=64, paths=[valid_log_path])

# Training phase
history_object = model.fit_generator(
                                        train_generator,
                                        samples_per_epoch=32256,
                                        nb_epoch=25,
                                        validation_data=valid_generator,
                                        nb_val_samples=1664  # , callbacks=[earlystopping_cb]

                                    )

# save model
model.save("model_nvidia_bt_512_e_25_resize_60_200_lr_5e-4_s_32256.h5")

# plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

