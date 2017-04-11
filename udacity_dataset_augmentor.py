import numpy as np
import csv
import random
from sklearn.model_selection import train_test_split

log_path = r'/home/mohamed/Udacity/CarND-Term1-Starter-Kit/Projects/CarND-Behavioral-Cloning-P3/Udacity_data/driving_log.csv'
img_path_prefix = r'/home/mohamed/Udacity/CarND-Term1-Starter-Kit/Projects/CarND-Behavioral-Cloning-P3/Udacity_data/'


train_log_path = r'/home/mohamed/Udacity/CarND-Term1-Starter-Kit/Projects/CarND-Behavioral-Cloning-P3/train_log.csv'
valid_log_path = r'/home/mohamed/Udacity/CarND-Term1-Starter-Kit/Projects/CarND-Behavioral-Cloning-P3/valid_log.csv'

steering_correction_value = 0.15
x = []
with open(log_path) as log_csv:
    log_reader = csv.reader(log_csv)

    for line in log_reader:
        #read center Steering Angle from file and apply correction value for  steering corrosponding to left and right camera images

        center_steering = float(line[3])
        if center_steering == 0.0 and np.random.randint(0, 5, 1) == 0:
            left_steering = center_steering + steering_correction_value
            right_steering = center_steering - steering_correction_value

            center_image = img_path_prefix + line[0]
            left_image = img_path_prefix + line[1].replace(' ', '')
            right_image = img_path_prefix+line[2].replace(' ', '')

            # append augmented data in one array
            x.append((center_image,center_steering))
            x.append((left_image, left_steering))
            x.append((right_image, right_steering))

# appending Dataset of cuptured prefect driving
log_path_prefect = r'/home/mohamed/Udacity/CarND-Term1-Starter-Kit/Projects/CarND-Behavioral-Cloning-P3/Perfect_Driving/driving_log.csv'

with open(log_path_prefect) as log_csv:
    log_reader = csv.reader(log_csv)

    for line in log_reader:

        center_image = line[0]
        left_image = line[1]
        right_image = line[2]
        center_steering = float(line[3])
        left_steering = center_steering + steering_correction_value
        right_steering = center_steering - steering_correction_value

        # append augmented data in one array

        x.append((center_image, center_steering))
        x.append((left_image, left_steering))
        x.append((right_image, right_steering))

# appending Dataset of cuptured prefect driving
log_path_conter_lap = r'/home/mohamed/Udacity/CarND-Term1-Starter-Kit/Projects/CarND-Behavioral-Cloning-P3/counter_Lap/driving_log.csv'

with open(log_path_conter_lap) as log_csv:
    log_reader = csv.reader(log_csv)

    for line in log_reader:

        center_image = line[0]
        left_image = line[1]
        right_image = line[2]
        center_steering = float(line[3])
        left_steering = center_steering + 0.2
        right_steering = center_steering - 0.2

        # append augmented data in one array

        x.append((center_image, center_steering))
        x.append((left_image, left_steering))
        x.append((right_image, right_steering))


random.shuffle(x)

X_train, X_valid = train_test_split(
    x, test_size=0.2)

with open(train_log_path, 'w', newline='') as train_file:
    train_writer = csv.writer(train_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for sample in X_train:
        train_writer.writerow(sample)

with open(valid_log_path, 'w', newline='') as valid_file:
    valid_writer = csv.writer(valid_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for sample in X_valid:
        valid_writer.writerow(sample)



