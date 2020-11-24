# -*- coding: utf-8 -*-

import numpy as np
from sklearn.linear_model import Perceptron
from data_formatter import DataFormatter
import decoupage_donnees as dd
import pandas as pd

class PerceptronClassifier:
    
    def __init__(self, lamb, x_train, t_train, x_test, t_test):
        
        print("Initialisation du perceptron...")
        
        #initialisation des données
        self.x_train = np.array(x_train)
        self.t_train = np.array(t_train)
        self.x_test = np.array(x_test)
        self.t_test = np.array(t_test)
        
        print("x_train shape : ", self.x_train.shape)
        print("t_train shape : ", self.t_train.shape)
        #print(self.t_train)
        print("x_test shape : ", self.x_test.shape)
        print("t_test shape : ", self.t_test.shape)
        #initialisation des poids
        self.w_0 = 1.
        self.w = np.zeros(self.x_train.shape[1])
        self.lamb = lamb
        
        self.classifier = Perceptron(eta0=self.lamb)
        
    """entrainement"""
    def entrainement(self):
        
        print("Entrainement...")
        self.classifier.fit(self.x_train, self.t_train)
        
        self.w_0 = np.array(self.classifier.intercept_[0])
        self.w = np.array(self.classifier.coef_[0])
        
        #self.test(x_train, t_train)
    
    
    def predict(self, x):
        
        dis = self.w_0 + np.dot(self.w.T, x)
        return np.argmax(dis)
    
    
    def test(self):
        nb_err = 0
        for i in range(self.x_test.shape[0]):
            prediction = self.predict(self.x_test[i])
            if(prediction != self.t_test[i]):
                nb_err += 1
                
        print("Pourcentage d'erreur de test : ", nb_err/self.x_test.shape[0])
        

   # def pourcentage_erreur(self, prediction, cibles):
        
        
#def __main__():
    
print("main test")
#lecture des données
print("=========================lecture des données") 
train = pd.read_csv('./train.csv', header=None)
train = np.array(train)



print("=========================formattage")
formatter = DataFormatter(train)
X, t = formatter.getDataAndTarget(False)
#print(t_train)

spliter = dd.DecoupeDonnees(X, t, 0.2)
[X_train, t_train, X_test, t_test] = spliter.mise_en_forme_donnees()


print("=========================Init percaptron")
perceptron = PerceptronClassifier(0.01, X_train, t_train, X_test, t_test)
perceptron.entrainement()
print("=========================test")
#perceptron.test()
print(perceptron.classifier.score(X_test,t_test))
    
    