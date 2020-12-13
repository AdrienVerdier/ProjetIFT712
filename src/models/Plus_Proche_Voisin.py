# -*- coding: utf-8 -*-

#####
# Félix Gaucher (gauf2611)
# Adrien Verdier (vera2704)
###

import numpy as np
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import random

class PlusProcheVoisin:
    def __init__(self, nb_voisin=5, weights='uniform', algorithm='auto', ppv=None):
        """
        Classe effectuant de la classification de nos données dans les classes de résultats à l'aide de la méthode
        des K plus proches voisins (KNeighborsClassifier)

        nb_voisin: le nombre de voisin à prendre comme paramètre
        weights: Si les poids sont uniformes ou s'ils dépendent de la distance entre les points
        algorithm: algorithm utilisé par le noyau (auto, ball_tree, kd_tree, brute)
        ppv: un modèle entrainé de plus proches voisins
        """
        self.nb_voisin = nb_voisin
        self.weights = weights
        self.algorithm = algorithm
        self.ppv = ppv

    def entrainement(self, x_train, y_train):
        """
        Entraîne une méthode d'apprentissage du type K plus proches voisins
        (KNeighborsClassifier). La variable x_train contient les entrées
        (une matrice numpy avec tous nos éléments d'entrainement) et des
        cibles t_train (un tableau 1D Numpy qui contient les cibles des
        éléments d'entrainemnet).
        """
        ppv = KNeighborsClassifier(n_neighbors=self.nb_voisin, weights=self.weights, algorithm=self.algorithm)
        ppv.fit(x_train, y_train)
        self.ppv = ppv

    def prediction(self, x):
        """
        Retourne la prédiction pour une entrée representée par un tableau
        1D Numpy ``x``.

        Cette méthode suppose que la méthode ``entrainement()`` a préalablement
        été appelée.

        Cette méthode va nous renvoyer la prédiction pour notre entrée ``x``,
        donc le numéro de la classe à laquelle l'élément appartient selon notre
        algorithme
        """

        return self.ppv.predict(x)

    def erreur(self, y, prediction):
        """
        Retourne la différence au carré entre
        la cible ``y`` et la prédiction ``prediction``.
        """
        return np.power(prediction - y, 2)

    def validation_croisee(self, x_tab, y_tab):
        """
        Cette fonction trouve les meilleurs hyperparametres ``self.nb_voisin``,
        ``self.weights`` et ``self.algorithm`` avec une validation croisée de
        type "k-fold" où k=5 avec les données contenues dans x_tab et y_tab.
        Une fois les meilleurs hyperparamètres trouvés, le modèle est entraîné
        une dernière fois.
        """

        plusProcheVoisin = KNeighborsClassifier()
        parameters = {'algorithm':('auto', 'ball_tree', 'kd_tree', 'brute'), 'weights' : ('uniform', 'distance'), 'n_neighbors': np.linspace(1, 5, 5, endpoint=True, dtype=np.int16)}
        self.ppv = GridSearchCV(plusProcheVoisin, parameters)
        self.ppv.fit(x_tab, y_tab)

        self.algorithm = self.ppv.best_estimator_.get_params()["algorithm"]
        self.weights = self.ppv.best_estimator_.get_params()["weights"]
        self.n_neighbors = self.ppv.best_estimator_.get_params()["n_neighbors"]
