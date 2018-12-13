import csv
import numpy as np
from scipy.stats import norm as normal
import sklearn
from sklearn import neighbors
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde as kde
from matplotlib.colors import Normalize
from matplotlib import cm
import networkx as nx
from random import randrange
from sklearn.metrics.pairwise import rbf_kernel
from math import cos, sin, pi, atan2, sqrt
from abc import ABC, abstractmethod

# import graph data
graph_data = []
import csv

class Regressor(ABC):
  @abstractmethod
  def fit(self, X, y):
    pass

  @abstractmethod
  def predict(self, X):
     pass


class NotYetTrainedException(Exception):
  """Learner must be trained (fit) before it can predict points."""
  pass

class AdaptiveLinearRegression(Regressor):
  def __init__(self, kernel, predictions):  # note kernel used in totally different way
    self.kernel = kernel
    self.X = None
    self.y = None
    self.predictions = predictions

  def fit(self, X, y):
    self.X = X
    self.y = y

  def predict(self, X):
    if self.X is not None:
      num_rows, num_cols = self.X.shape
      X_K = np.zeros((num_rows, num_cols))

      num_test_points = X.shape[0]

      predictions = []

      for test_point in range(num_test_points):
        for row in range(num_rows):
          for col in range(num_cols):
            X_K[row, col] = self.kernel(self.predictions[test_point], self.y[row]) * self.X[row, col]

        A = np.dot(X_K.T, self.X)
        b = np.dot(X_K.T, self.y)

        theta = np.linalg.solve(A, b)

        prediction = np.dot(X[test_point,], theta)
        predictions.append(prediction)

      prediction_vector = np.array(predictions)
      prediction_vector = prediction_vector.reshape(-1, 1)
      return prediction_vector
    else:
      raise NotYetTrainedException
    pass


#Define some matrices to hold the data
X_tr = []
y_tr_lat = []
y_tr_lon = []
X_te = []
graph_preds_lat = []
graph_preds_lon = []
ids_te = []
numerical_data = [X_tr, X_te]

with open("posts_train.txt", mode = 'r') as csv_file:
  csv_reader = csv.DictReader(csv_file)
  for row in csv_reader:
    if(float(row['Hour1']) != 25 and float(row['Hour2']) != 0 and float(row['Hour3']) != 0): #ignore people who don't post often
      if(float(row['Lat']) != 0.0 or float(row['Lon']) != 0.0): #ignore people on the Null Island
        X_tr.append([float(row['Hour1']), float(row['Hour2']), float(row['Hour3']), float(row['Posts'])])
        y_tr_lat.append(float(row['Lat']))
        y_tr_lon.append(float(row['Lon']))

with open('posts_test.txt', mode='r') as csv_file:
  csv_reader = csv.DictReader(csv_file)
  for row in csv_reader:
    #If the user doesn't have an nth most frequent posting hour, change it to something reasonable (i.e the next most frequent hour of posting)
    hour_1 = int(row['Hour1'])
    hour_2 = int(row['Hour2'])
    hour_3 = int(row['Hour3'])

    if(hour_2 == 25):
      hour_2 = hour_1
      hour_3 = hour_1
    elif(hour_3 == 25):
      hour_3 = hour_2

    X_te.append([float(row['Hour1']), float(hour_2), float(hour_3), float(row['Posts'])])

with open('answer.txt', mode = 'r') as csv_file:
  csv_reader = csv.DictReader(csv_file)
  for row in csv_reader:
    graph_preds_lat.append(float(row['Lat']))
    graph_preds_lon.append(float(row['Lon']))
    ids_te.append(float(row['Id']))

#convert data to numpy arrays
#for d in numerical_data:
#  d = np.array(d)
X_tr = np.array(X_tr)
X_te = np.array(X_te)

#Mess of kernel functions
def latitude_kernel(mean, test_value):
  stdev = 10 #NOT set in stone
  extremity_value = np.absolute(mean - test_value)

  return normal_cdf(extremity_value, stdev)

def longitude_kernel(mean, test_value):
  #machinery of spherical coordinates overkill in this case
  stdev = 15 #NOT set in stone
  distance = np.absolute(mean - test_value)
  anti_distance = np.absolute(360 - distance)
  extremity_value = np.minimum(distance, anti_distance)

  return normal_cdf(extremity_value, stdev)

def normal_cdf(test_value, stdev):
  prob = normal.sf(test_value, loc = 0, scale = stdev)
  prob *= 2
  return prob

#Now onto the regression
latitude_regressor = AdaptiveLinearRegression(kernel = latitude_kernel(), predictions = graph_preds_lat)
latitude_regressor.fit(X_tr, y_tr_lat)
lat_predictions = latitude_regressor.predict(X_te)

longitude_regressor = AdaptiveLinearRegression(kernel = longitude_kernel(), predictions = graph_preds_lon)
longitude_regressor.fit(X_tr, y_tr_lon)
lon_predictions = longitude_regressor.predict(X_te)

result = []
labels = ['Id', 'Lat', 'Lon']
result.append(labels)

for i in range(len(lat_predictions)): #check range is good here
  point = []
  point.append(ids_te[i])
  point.append(lat_predictions[i])
  point.append(lon_predictions[i])

with open('adaptive_answer.txt', "w") as output:
  writer = csv.writer(output, lineterminator = '\n')
  writer.writerows(result)



