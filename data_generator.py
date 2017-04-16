import numpy as np
import cv2
import csv
from matplotlib import pyplot as plt
import random


def dataset_generator(batch_size=32, paths=[]):
    image_width = 200
    image_height = 66
    image_channels = 3

    # Load Dataset Paths to memory to facilitate interaction between generator and dataset
    for path in paths:
        dataset = []
        with open(path) as log_csv:
            log_reader = csv.reader(log_csv)
            for line in log_reader:
                image = line[0]
                steering = float(line[1])
                dataset.append((image, steering))


    while 1:
        # create data placeholders with given batch size
        x = np.zeros((batch_size, image_height, image_width, image_channels),dtype=np.uint8)
        y = np.zeros(batch_size)
        # Shuffling Dataset every epoch
        random.shuffle(dataset)
        i = 0
        for data_sample in dataset:
                try:
                    line = data_sample
                    img_path = (line[0])
                    flip_flage = np.random.choice([-1,1], 1)
                    image = cv2.imread(img_path)[..., ::-1]
                    image_cropped = image[60:(160-25),:]
                    image_resized = cv2.resize(image_cropped, (200, 66), interpolation=cv2.INTER_AREA)
                    if flip_flage == 1:

                        x[i, :, :, :] = image_resized
                        y[i] = float(line[1])
                    else:
                        x[i, :, :, :] = np.fliplr(image_resized)
                        y[i] = -1 * float(line[1])
                    i += 1
                    if i == batch_size:
                        i = 0
                        yield x, y
                except:
                    pass



# Generator Testing script

if __name__ == '__main__':

    data_generator = dataset_generator(batch_size=32, paths=[r'/home/mohamed/Udacity/CarND-Term1-Starter-Kit/Projects/CarND-Behavioral-Cloning-P3/train_log.csv'])
    for i in range(2):
        x, y = next(data_generator)
        plt.imshow(np.fliplr(x[10]))
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()