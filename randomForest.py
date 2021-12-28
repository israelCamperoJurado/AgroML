# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 21:08:42 2021

@author: 20210595
"""


from agroml.utils.splitDataByYear import splitDataByYear
from agroml.models.mlpRegression import MultiLayerPerceptron
from agroml.models.transformerLstmRegression import transformerLSTM
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from agroml.models.rfRegression import RandomForest
from agroml.utils.statistics import *
from agroml.utils.plots import *
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

if __name__ == '__main__':
    # import dataset from example
    df = pd.read_csv('tests/test-data/regression_dataset.csv', names = ['distance_P', 
                                                                        'distance_Q', 
                                                                        'distance_S', 
                                                                        'distance_T', 
                                                                        'distance_P_onset', 
                                                                        'distance_T_offset', 
                                                                        'target'])
    size=np.array(df.shape)
    X = df.iloc[:, 0:(size[1]-1)].values  
    y = df.iloc[:, (size[1]-1)].values 
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    y_train = y_train.reshape(len(y_train), 1) 
    y_test = y_test.transpose().reshape(len(y_test), 1)
    
    mlModel = RandomForest(X_train, X_test, y_train, y_test)
    # Hiperparameter optimization using Bayesian optimization
    mlModelBayes, bestParams = mlModel.bayesianOptimization()
    
    # train best model with the full dataset
    mlModel.trainFullTrainingData(mlModelBayes, showGraph=False)
    y_pred = mlModel.predictFullTestingData(mlModelBayes)
    mbe = getMeanBiasError(y_test, y_pred)
    rmse = getRootMeanSquaredError(y_test, y_pred)
    nse = getNashSuteliffeEfficiency(y_test, y_pred)
    
    # plot predictions vs. measured
    plotGraphLinealRegresion(
        x = y_test, 
        xName = 'Measures values', 
        y = y_pred, 
        yName = 'Predicted values')