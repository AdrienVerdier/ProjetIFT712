# -*- coding: utf-8 -*-

#####
# Félix Gaucher (gauf2611)
# Adrien Verdier (vera2704)
###

import numpy as np
import sys
import random
from sklearn.linear_model import Perceptron
from sklearn.model_selection import GridSearchCV

class PerceptronModele:
    def __init__(self, penalty=None, alpha=0.0001, max_iter=1000, n_iter_no_change=5, pct=None):
        """
        Classe effectuant de la classification de nos données dans les classes de résultats à l'aide de la méthode
        du perceptron (Perceptron)

        penalty: Le terme de régularisation à utiliser
        alpha: Constante qui multiplie le terme de régularisation s'il est présent
        max_iter: Nombre maximum d'itération
        n_iter_no_change: nombre d'itération sans changement avant d'arrêter prématurément
        pct: Un modèle entrainé de perceptron
        """
        self.penalty = penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.n_iter_no_change = n_iter_no_change
        self.pct = pct

    def entrainement(self, x_train, y_train):
        """
        Entraîne une méthode d'apprentissage du type Perceptron
        (Perceptron). La variable x_train contient
        les entrées (une matrice numpy avec tous nos éléments d'entrainement)
        et des cibles t_train (un tableau 1D Numpy qui contient les cibles des
        éléments d'entrainemnet).
        """
        pct = Perceptron(penalty=self.penalty, alpha=self.alpha, max_iter=self.max_iter, n_iter_no_change=self.n_iter_no_change)
        pct.fit(x_train, y_train)
        self.pct = pct

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
        return self.pct.predict(x)

    def erreur(self, y, prediction):
        """
        Retourne la différence au carré entre
        la cible ``y`` et la prédiction ``prediction``.
        """
        return np.power(prediction - y, 2)

    def validation_croisee(self, x_tab, y_tab):
        """
        Cette fonction trouve les meilleurs hyperaparamètres ``self.penalty``,
        ``self.n_iter_no_change``, ``self.alpha``, ``self.max_iter``,
        avec une validation croisée de type "k-fold" où k=5 avec les
        données contenues dans x_tab et y_tab. Une fois le meilleur
        hyperparamètre trouvé, le modèle est entraîné une dernière fois.
        """

        classifier = Perceptron()
        # 'max_iter':np.linspace(500, 2500, 3, endpoint=True, dtype=np.int16)
        parameters = {'penalty':['l2','l1', 'elasticnet', None], 'alpha':np.linspace(0.001, 3, 5, endpoint=True), 'n_iter_no_change':np.linspace(3, 15, 3, endpoint=True, dtype=np.int16)}
        self.pct = GridSearchCV(classifier, parameters)
        self.pct.fit(x_tab, y_tab)

        self.penalty = self.pct.best_estimator_.get_params()["penalty"]
        self.alpha = self.pct.best_estimator_.get_params()["alpha"]
        self.max_iter = self.pct.best_estimator_.get_params()["max_iter"]
        self.n_iter_no_change = self.pct.best_estimator_.get_params()["n_iter_no_change"]
