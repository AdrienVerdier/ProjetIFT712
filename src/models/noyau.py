# -*- coding: utf-8 -*-

#####
# Félix Gaucher (gauf2611)
# Adrien Verdier (vera2704)
###

import numpy as np
import sys
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV
import random

class Noyau:
    def __init__(self, kernel="rbf", C=0.025, probability=True, svc=None):
        """
        Classe effectuant de la classification de nos données dans les classes de résultats à l'aide de la méthode
        du noyau (SVC)

        kernel: le noyau a utiliser
        C: Le paramètre de régularisation
        probability: Utilisation les estimations probabilistes
        svc : Un modèle SVC déjà entrainé
        """
        self.kernel = kernel
        self.C = C
        self.probability = probability
        self.svc = None

    def entrainement(self, x_train, y_train):
        """
        Entraîne une méthode d'apprentissage du type noyau
        (SVC). La variable x_train contient
        les entrées (une matrice numpy avec tous nos éléments d'entrainement)
        et des cibles t_train (un tableau 1D Numpy qui contient les cibles des
        éléments d'entrainemnet).
        """
        svc = SVC(kernel="rbf", C=0.025, probability=True)
        svc.fit(x_train, y_train)
        self.svc = svc

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
        return self.svc.predict(x)

    def erreur(self, y, prediction):
        """
        Retourne la différence au carré entre
        la cible ``y`` et la prédiction ``prediction``.
        """
        return np.power(prediction - y, 2)

    def validation_croisee(self, x_tab, y_tab):
        """
        Cette fonction trouve les meilleurs hyperparamètres ``self.kernel``,
        ``self.C``, ``self.probability``
        avec une validation croisée de type "k-fold" où k=5 avec les
        données contenues dans x_tab et y_tab. Une fois le meilleur
        hyperparamètre trouvé, le modèle est entraîné une dernière fois.
        """

        classifier = SVC()
        parameters = {'kernel':('linear', 'poly', 'rbf', 'sigmoid'), 'C':np.linspace(1, 15, 5, endpoint=True), 'probability':(True, False)}
        self.svc = GridSearchCV(classifier, parameters)
        self.svc.fit(x_tab, y_tab)

        self.kernel = self.svc.best_estimator_.get_params()["kernel"]
        self.C = self.svc.best_estimator_.get_params()["C"]
        self.probability = self.svc.best_estimator_.get_params()["probability"]