import sys
sys.path.insert(0, '..')

from utils import data
import os
import sklearn
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import shift
from sklearn.preprocessing import normalize

from gradient_descent import GradientDescent
from metrics import accuracy

# ------------ HYPERPARAMETERS -------------
BASE_PATH = '../COVID-19/csse_covid_19_data/'
# ------------------------------------------


""" 
I am interested in looking at how the rate of cases affects the death rate (deaths/cases)
I am expecting that higher jumps in the number of cases increases the death rate
I will use linear regression then a neural network to see if there is a relationship, and if so what it is. 
"""


"""
Loading data
SECTION 1 BELOW
"""

confirmed = os.path.join(
    BASE_PATH, 
    'csse_covid_19_time_series',
    'time_series_covid19_confirmed_global.csv')
confirmed = data.load_csv_data(confirmed)

dead = os.path.join(
    BASE_PATH, 
    'csse_covid_19_time_series',
    'time_series_covid19_deaths_global.csv')
dead = data.load_csv_data(dead)

confirmed_features = []
confirmed_targets = []
dead_features = []
dead_targets = []

for val in np.unique(confirmed["Country/Region"]):
    df = data.filter_by_attribute(
        confirmed, "Country/Region", val)
    cases, labels = data.get_cases_chronologically(df)
    confirmed_features.append(cases)
    confirmed_targets.append(labels)

for val in np.unique(dead["Country/Region"]):
    df = data.filter_by_attribute(
        dead, "Country/Region", val)
    cases, labels = data.get_cases_chronologically(df)
    dead_features.append(cases)
    dead_targets.append(labels)

confirmed_features = np.concatenate(confirmed_features, axis=0)
dead_features = np.concatenate(dead_features, axis=0)


""" 
STATISTICAL CHECK
"""

death_rate = np.divide(dead_features, confirmed_features, out=np.zeros_like(dead_features), where=confirmed_features!=0)

confirmed_polynomials = []
for example in confirmed_features:
    polynomial = np.polyfit(np.arange(example.shape[0]).astype(float), np.log(example.astype(float), out=np.ones_like(example).astype(float), where=example!=0), 1)
    confirmed_polynomials.append(polynomial)

growth_rates =  (np.array(confirmed_polynomials))[:, 0]
average_death_rate = np.average(death_rate, axis=1)

r = np.correlate(growth_rates, average_death_rate)
#print(r)

"""
GRADIENT DESCENT CHECK
"""

# split training and testing data, normalize features
average_death_rate = np.reshape(average_death_rate, (-1, 1))

# option 1: normalize features
#confirmed_features = normalize(confirmed_features, axis=1)

# option 2: binary inputs
gradients = np.gradient(confirmed_features)
confirmed_features = -np.sign(gradients[0])
print(confirmed_features.shape)

data = np.hstack((confirmed_features, average_death_rate))
np.random.shuffle(data)
training_data = data[0:212,:]
testing_data = data[212:, :]
training_features = training_data[:, :-1]
training_targets = training_data[:, -1]
testing_features = testing_data[:, :-1]
testing_targets = testing_data[:, -1]

model = GradientDescent('squared',  regularization='l1', reg_param=.5)
model.fit(training_features.astype(float), training_targets.astype(float))
prediction = model.confidence(testing_features.astype(float))

error = ((testing_targets - prediction)**2).mean()

r = np.correlate(prediction, testing_targets)
print(r)