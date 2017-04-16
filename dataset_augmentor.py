import numpy as np
import csv
import random
from sklearn.model_selection import train_test_split



train_log_path = r'/home/mohamed/Udacity/CarND-Term1-Starter-Kit/Projects/CarND-Behavioral-Cloning-P3/train_log.csv'
valid_log_path = r'/home/mohamed/Udacity/CarND-Term1-Starter-Kit/Projects/CarND-Behavioral-Cloning-P3/valid_log.csv'

# Dataset Augmentation by using three camera images to simulate bad examples to network.
steering_correction_value = 0.15


# data set container
x = []

# appending Dataset provided by udacity
# this data set need special handling to get abs. path of images
log_path = r'/home/mohamed/Udacity/CarND-Term1-Starter-Kit/Projects/CarND-Behavioral-Cloning-P3/Udacity_data/driving_log.csv'
img_path_prefix = r'/home/mohamed/Udacity/CarND-Term1-Starter-Kit/Projects/CarND-Behavioral-Cloning-P3/Udacity_data/'

with open(log_path) as log_csv:
    log_reader = csv.reader(log_csv)

    for line in log_reader:
        #read center Steering Angle from file and apply correction value for  steering corrosponding to left and right camera images

        center_steering = float(line[3])
        if center_steering == 0.0:
            if np.random.randint(0, 10, 1) == 0:
                left_steering = center_steering + steering_correction_value
                right_steering = center_steering - steering_correction_value

                center_image = img_path_prefix + line[0]
                left_image = img_path_prefix + line[1].replace(' ', '')
                right_image = img_path_prefix+line[2].replace(' ', '')

                # append augmented data in one array
                x.append((center_image,center_steering))
                x.append((left_image, left_steering))
                x.append((right_image, right_steering))
        else:
            left_steering = center_steering + steering_correction_value
            right_steering = center_steering - steering_correction_value

            center_image = img_path_prefix + line[0]
            left_image = img_path_prefix + line[1].replace(' ', '')
            right_image = img_path_prefix + line[2].replace(' ', '')

            # append augmented data in one array
            x.append((center_image, center_steering))
            x.append((left_image, left_steering))
            x.append((right_image, right_steering))


# appending Dataset of captured prefect driving
log_path_prefect = r'/home/mohamed/Udacity/CarND-Term1-Starter-Kit/Projects/CarND-Behavioral-Cloning-P3/Perfect_Driving/driving_log.csv'
# appending Dataset of captured prefect driving
log_path_conter_lap = r'/home/mohamed/Udacity/CarND-Term1-Starter-Kit/Projects/CarND-Behavioral-Cloning-P3/counter_Lap/driving_log.csv'
# appending Dataset of captured corners driving
log_path_corners_lap = r'/home/mohamed/Udacity/CarND-Term1-Starter-Kit/Projects/CarND-Behavioral-Cloning-P3/corners_Data/driving_log.csv'

log_pathes = [log_path_prefect,log_path_conter_lap,log_path_corners_lap]
for path in log_pathes:
    with open(path) as log_csv:
        log_reader = csv.reader(log_csv)

        for line in log_reader:

            center_steering = float(line[3])
            if center_steering == 0.0:
                if np.random.randint(0, 10, 1) == 0:
                    left_steering = center_steering + steering_correction_value
                    right_steering = center_steering - steering_correction_value

                    center_image = line[0]
                    left_image = line[1].replace(' ', '')
                    right_image = line[2].replace(' ', '')

                    # append augmented data in one array
                    x.append((center_image, center_steering))
                    x.append((left_image, left_steering))
                    x.append((right_image, right_steering))
            else:
                left_steering = center_steering + steering_correction_value
                right_steering = center_steering - steering_correction_value

                center_image = line[0]
                left_image = line[1].replace(' ', '')
                right_image = line[2].replace(' ', '')

                # append augmented data in one array
                x.append((center_image, center_steering))
                x.append((left_image, left_steering))
                x.append((right_image, right_steering))

random.shuffle(x)

# Split dataset to training and validation but mainly testing done by using realtime processing with simulator
X_train, X_valid = train_test_split(
    x, test_size=0.05)

with open(train_log_path, 'w', newline='') as train_file:
    train_writer = csv.writer(train_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for sample in X_train:
        train_writer.writerow(sample)

with open(valid_log_path, 'w', newline='') as valid_file:
    valid_writer = csv.writer(valid_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for sample in X_valid:
        valid_writer.writerow(sample)



