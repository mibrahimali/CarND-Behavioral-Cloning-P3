import numpy as np
import csv
import matplotlib.pyplot as plt

train_log_path = r'/home/mohamed/Udacity/CarND-Term1-Starter-Kit/Projects/CarND-Behavioral-Cloning-P3/train_log.csv'

valid_log_path = r'/home/mohamed/Udacity/CarND-Term1-Starter-Kit/Projects/CarND-Behavioral-Cloning-P3/valid_log.csv'


training_steering = []
with open(train_log_path) as log_csv:
    log_reader = csv.reader(log_csv)
    for line in log_reader:
        training_steering.append(float(line[1]))

validation_steering = []
with open(valid_log_path) as log_csv:
    log_reader = csv.reader(log_csv)
    for line in log_reader:
        validation_steering.append(float(line[1]))


training_dataset_size = len(training_steering)
validation_dataset_size = len(validation_steering)

print('Training Dataset size =', training_dataset_size)
print('Validation Dataset size =', validation_dataset_size)

plt.figure()
plt.hist(training_steering, bins=100)
plt.title("Training dataset Steering command Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")

plt.figure()
plt.hist(validation_steering, bins=100)
plt.title("Validation dataset Steering command Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")



plt.show()