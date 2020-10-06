# -*- coding: utf-8 -*-

### IMPORTS
import numpy as np
import pandas as pd

"""
Permet de mettre les données dans un tableau numpy

param : data (données brut en .csv)
return : données formatées 
"""
def formatage(data):
    
    data = data.to_numpy()
    return data


"""
Recupère toutes les etiquettes de classes dans le fichier approprié
(ici : sample_submission.csv)

param : dataPath (chemin vers les données)
return : liste de toutes les étiquettes de classe
"""
def getAllEtiquettes(dataPath):
    
    sample = pd.read_csv(dataPath, header=None)
    sample = formatage(sample)
    
    return sample[0][1:]

"""
Construction des cibles pour la phase d'entrainement

param : trainData (données d'entrainement)
return : tableau des cibles (one-hot encoding)
"""
def constructionCibles(trainData):
    
    #recuperation de toutes les étiquettes de classes
    etiquettes = getAllEtiquettes('../../data/raw/sample_submission.csv')
    
    