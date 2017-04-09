import numpy as np
import csv
import random
from sklearn.model_selection import train_test_split

log_path = r'/home/mohamed/Udacity/CarND-Term1-Starter-Kit/Projects/CarND-Behavioral-Cloning-P3/Udacity_data/driving_log.csv'
img_path_prefix = r'/home/mohamed/Udacity/CarND-Term1-Starter-Kit/Projects/CarND-Behavioral-Cloning-P3/Udacity_data/'


train_log_path = r'/home/mohamed/Udacity/CarND-Term1-Starter-Kit/Projects/CarND-Behavioral-Cloning-P3/Udacity_data/train_log.csv'
valid_log_path = r'/home/mohamed/Udacity/CarND-Term1-Starter-Kit/Projects/CarND-Behavioral-Cloning-P3/Udacity_data/valid_log.csv'

x = []
with open(log_path) as log_csv:
    log_reader = csv.reader(log_csv)

    for line in log_reader:

        center_image = img_path_prefix + line[0]
        left_image = img_path_prefix + line[1].replace(' ','')
        right_image = img_path_prefix+line[2].replace(' ','')
        center_steering = float(line[3])
        left_steering = center_steering - 0.2
        right_steering = center_steering + 0.2

        # append augmented data in one array

        x.append((center_image,center_steering))
        x.append((left_image, left_steering))
        x.append((right_image, right_steering))

random.shuffle(x)

X_train, X_valid = train_test_split(
    x, test_size=0.2, random_state=42)

with open(train_log_path, 'w', newline='') as train_file:
    train_writer = csv.writer(train_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for sample in X_train:
        train_writer.writerow(sample)

with open(valid_log_path, 'w', newline='') as valid_file:
    valid_writer = csv.writer(valid_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for sample in X_valid:
        valid_writer.writerow(sample)
