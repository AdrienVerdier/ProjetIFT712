# -*- coding: utf-8 -*-

import numpy as np

class DataFormatter:
    
    def __init__(self, dataset):
        """
        Initialisation du formatteur de données

        Parameters
        ----------
        dataset : np.array
            jeu de données à traiter

        Returns
        -------
        None.

        """
        self.dataset = dataset
        
        
    def getLabel(self):
        """
        Associe chaque étiquette de classe à un entier naturel afin de
        faciliter la création des one hot vector

        Parameters
        ----------
        data : jeu de données
            DESCRIPTION.

        Returns
        -------
        labels : np array
            [labels, nombre associé]
        """
        data = self.dataset
        nbClasse = 0
        t_all = data[0:,1]
        labels = []
        
        
        for i in range(t_all.shape[0]):    
            #si l'etiquette de classe est deja dedans
            if(i == 0 or np.isin(t_all[i], labels, True)):
                continue
            
            label = []
            label.append(t_all[i])
            label.append(nbClasse)
            labels.append(label)
            nbClasse += 1
                 
        return labels
    
    
    def getDataAndTarget(self):
        """

        Parameters
        ----------
        data : TYPE
            DESCRIPTION.

        Returns
        -------
        X : données d'entrainement/test
            list contenant la liste de tous les éléments
        t : cibles associées aux données d'entrainement
            format : one-hot-vector

        """
        data = self.dataset
        data = np.array(data)
        X = []
        t = []
        labels = np.array(self.getLabel(data))
        
        #pour chaque donnée
        for i in range(1,data.shape[0]):
            
            #creation des x_train
            x = np.array(data[i, 2:])
            x = x.astype(np.float)
            x = x.tolist()
            X.append(x)
            
            #creation des t_train
            one_hot_vector = np.zeros(labels.shape[0])
            
            t_strings = data[i, 1]
            label_place = np.where(labels == t_strings)
            label = int(labels[label_place[0], 1])
            
            one_hot_vector[label] = 1.0
            one_hot_vector = one_hot_vector.tolist()
            t.append(one_hot_vector)
    
    
        return X, t
    
    
    
    
    
    
    
    
    
    
    
    
    