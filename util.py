import pandas as pd
import numpy as np
import random
import csv

def generate_1d_dataset(N, variance=100):
    X = np.matrix(range(N)).T + 1
    Y = np.matrix([random.random() * variance + i * 10 + 900 for i in range(len(X))]).T
    return X, Y

def load_dataset_csv(x_path, y_path):
    x_file = open(x_path, 'rb')
    y_file = open(y_path, 'rb')
    
    x_csv_reader = csv.reader(x_file, delimiter=',')
    y_csv_reader = csv.reader(y_file, delimiter=',')

    xs = np.array([x for x in x_csv_reader], dtype=np.float32)
    ys = np.array([y for y in y_csv_reader], dtype=np.float32)

    return xs, ys

