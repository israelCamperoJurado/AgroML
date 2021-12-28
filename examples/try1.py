# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 16:20:45 2021

@author: 20210595
"""

from agroml.utils.splitDataByYear import splitDataByYear
from agroml.models.mlpRegression import MultiLayerPerceptron
from agroml.utils.statistics import *
from agroml.utils.plots import *
import pandas as pd

if __name__ == '__main__':
   

  # import dataset from example
  df = pd.read_csv('tests/test-data/data-example.csv', sep=';')
  print(df)

  # get important variables
  uniqueStations = np.unique(df['station'])
  uniqueYears = np.unique(df['year'])
  varListInputs = ['tx', 'tn', 'rs', 'day']
  varListOutputs = ['et0']

  # split data to train and test
  xTrain, xTest, yTrain, yTest = splitDataByYear(
      df=df,
      station=uniqueStations[-1], 
      yearTestStart=uniqueYears[-3], 
      varListInputs=varListInputs, 
      varListOutputs=varListOutputs,
      preprocessing = 'standardization')
  print(xTrain)
  print(yTrain)