import data_generator
import models
from keras.optimizers import Adam
# from keras.utils.visualize_util import plot


lenet_model = models.lenet_model((160, 320, 3))
optimizer = Adam(lr=1e-4)
lenet_model.compile(optimizer, loss="mse")
# plot(lenet_model, to_file='model.png')


train_generator = data_generator.dataset_generator(batch_size=16, paths=[r'C:\Storage\Udacity\CarND-Term1-Starter-Kit\Projects\CarND-Behavioral-Cloning-P3\data_set_counter_lap\driving_log.csv'])

lenet_model.fit_generator(
    train_generator,
    samples_per_epoch=2700,
    nb_epoch=5,

)

lenet_model.save("lenet_model.h5")