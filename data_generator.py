import numpy as np
import cv2
import csv
from matplotlib import pyplot as plt


def dataset_generator(batch_size=32, paths=[]):
    image_width = 320
    image_height = 160
    image_channels = 3
    while 1:
        # create data placeholders with given batch size
        x = np.zeros((batch_size, image_height, image_width, image_channels),dtype=np.uint8)
        y = np.zeros(batch_size)
        csv_readers = []

        for path in paths:
            csv_readers.append(csv.reader(open(path)))
        i = 0
        for reader_id, csv_reader in enumerate(csv_readers):
            # for line in csv_reader:
            while csv_reader:
                try:
                    line = next(csv_reader)
                    x[i, :, :, :] = cv2.imread((line[0]).replace('\\', '/'))[..., ::-1]
                    y[i] = float(line[3])
                    i += 1
                    if i == batch_size:
                        i = 0
                        yield x, y
                except StopIteration:
                    break



# Generator Testing script

if __name__ == '__main__':
    data_generator = dataset_generator(batch_size=32, paths=[r'C:\Storage\Udacity\CarND-Term1-Starter-Kit\Projects\CarND-Behavioral-Cloning-P3\data_set_counter_lap\driving_log.csv'])
    for i in range(10):
        x, y = next(data_generator)
        plt.imshow(x[0])
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()