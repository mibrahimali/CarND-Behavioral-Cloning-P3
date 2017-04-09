import numpy as np
import cv2
import csv
from matplotlib import pyplot as plt
import random

def dataset_generator(batch_size=32, paths=[]):
    image_width = 320
    image_height = 160
    image_channels = 3
    while 1:
        # create data placeholders with given batch size
        x = np.zeros((batch_size, image_height, image_width, image_channels),dtype=np.uint8)
        y = np.zeros(batch_size)
        csv_readers = []
        # adding shuffling step of data

        for path in paths:
            file = []
            with open(path) as log_csv:
                log_reader = csv.reader(log_csv)
                for line in log_reader:
                    image = line[0]
                    steering = float(line[1])
                    file.append((image, steering))
            random.shuffle(file)
            with open(path, 'w', newline='') as suffled_file:
                file_writer = csv.writer(suffled_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                print("writing shuffled data")
                for sample in file:
                    file_writer.writerow(sample)

        for path in paths:
            csv_readers.append(csv.reader(open(path)))
        i = 0
        for reader_id, csv_reader in enumerate(csv_readers):
            # for line in csv_reader:
            while csv_reader:
                try:
                    line = next(csv_reader)
                    img_path = (line[0])
                    flip_flage = np.random.choice([-1,1], 1)
                    if flip_flage == 1:
                        x[i, :, :, :] = cv2.imread(img_path)[..., ::-1]
                        y[i] = float(line[1])
                    else:
                        x[i, :, :, :] = np.fliplr(cv2.imread(img_path)[..., ::-1])
                        y[i] = -1 * float(line[1])
                    i += 1
                    if i == batch_size:
                        i = 0
                        yield x, y
                except StopIteration:
                    break



# Generator Testing script

if __name__ == '__main__':
    data_generator = dataset_generator(batch_size=32, paths=[r'/home/mohamed/Udacity/CarND-Term1-Starter-Kit/Projects/CarND-Behavioral-Cloning-P3/Udacity_data/train_log.csv'])
    # data_generator = dataset_generator(batch_size=32, paths=[r'C:\Storage\Udacity\CarND-Term1-Starter-Kit\Projects\CarND-Behavioral-Cloning-P3\data_set_counter_lap\driving_log.csv'])
    for i in range(10):
        x, y = next(data_generator)
        plt.imshow(x[0])
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()