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
import numpy as np

if __name__ == '__main__':
   

    # import dataset from example
    df = pd.read_csv('tests/test-data/data-example.csv', sep=';')
    # get important variables
    uniqueStations = np.unique(df['station'])
    uniqueYears = np.unique(df['year'])
    varListInputs = ['tx', 'tn', 'rs', 'day']
    varListOutputs = ['et0']
    
    # split data to train and test
    xTrain, xTest, yTrain, yTest, scaler= splitDataByYear(
        df=df,
        station=uniqueStations[-1], 
        yearTestStart=uniqueYears[-3], 
        varListInputs=varListInputs, 
        varListOutputs=varListOutputs,
        preprocessing = 'standardization'
        )
    
    # create model
    mlModel = MultiLayerPerceptron(xTrain, xTest, yTrain, yTest)
    # Hiperparameter optimization using Bayesian optimization
    mlModelBayes, bestParams = mlModel.bayesianOptimization(
        hiddenLayersList=[1,2], 
        neuronsList=[1, 20], 
        activationList=['relu'], 
        optimizerList=['adam'], 
        epochsList=[50,100], 
        bayesianEpochs=5, 
        randomStart=4, 
        validationSplit=0.2, 
        shuffle=False)
    
    # train best model with the full dataset
    mlModel.trainFullTrainingData(mlModelBayes, showGraph=False)
    yPred = mlModel.predictFullTestingData(mlModelBayes)
    mbe = getMeanBiasError(yTest, yPred)
    rmse = getRootMeanSquaredError(yTest, yPred)
    nse = getNashSuteliffeEfficiency(yTest, yPred)
    
    # plot predictions vs. measured
    plotGraphLinealRegresion(
        x = yTest, 
        xName = 'Measures values', 
        y = yPred, 
        yName = 'Predicted values')