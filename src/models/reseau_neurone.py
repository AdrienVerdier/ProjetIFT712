# -*- coding: utf-8 -*-

#####
# Félix Gaucher (gauf2611)
# Adrien Verdier (vera2704)
###

import numpy as np
import sys
import random
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

class ReseauNeurone:
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001, learning_rate='constant',
                    learning_rate_init=0.001, power_t=0.5, max_iter=200, momentum=0.9, 
                    beta_1=0.9, beta_2=0.999, n_iter_no_change=10, max_fun=15000, rn=None):
        """
        Classe effectuant de la classification de nos données dans les classes de résultats à l'aide de la méthode
        du réseau de neurone multicouche (MLPClassifier -> MultiLayerPerceptron)

        hidden_layer_sizes: Nombre de neurone par couche caché
        activation: Fonction d'activation pour les couches cachés
        solver: Solver pour l'optimisation des poids
        alpha: Therme de régularisation
        learning_rate: le learning rate a utiliser pour l'optimisation des poids
        learning_rate_init: le learning rate utilisé au départ
        power_t: l'exposant pour le learning rate inversé, utile pour le solver sgd
        max_iter: nombre d'iteération maximum
        momentum: momentum pour la descente de gradient, pour sgd
        beta_1: le decay rate pour l'estimation du premier vecteur de moment (adam)
        beta_2: le decay rate pour l'estimation du deuxième vecteur de moment (adam)
        n_iter_no_change: nombre d'itération sans changement avant d'arrêter prématurément
        max_fun: nombre maximum d'appelle a la fonction de loss dans 'lbfgs'
        rn: un modèle entrainé de réseau de neurone
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.power_t = power_t
        self.max_iter = max_iter
        self.momentum = momentum
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.n_iter_no_change = n_iter_no_change
        self.max_fun = max_fun
        self.rn = None

    def entrainement(self, x_train, y_train):
        """
        Entraîne une méthode d'apprentissage du type réseau de neurones multicouches
        (MLPClassifier). La variable x_train contient
        les entrées (une matrice numpy avec tous nos éléments d'entrainement)
        et des cibles t_train (un tableau 1D Numpy qui contient les cibles des
        éléments d'entrainemnet).
        """
        rn = MLPClassifier(hidden_layer_sizes = self.hidden_layer_sizes,
                activation = self.activation,
                solver = self.solver,
                alpha = self.alpha,
                learning_rate = self.learning_rate,
                learning_rate_init = self.learning_rate_init,
                power_t = self.power_t,
                max_iter = self.max_iter,
                momentum = self.momentum,
                beta_1 = self.beta_1,
                beta_2 = self.beta_2,
                n_iter_no_change = self.n_iter_no_change,
                max_fun = self.max_fun)
        rn.fit(x_train, y_train)
        self.rn = rn

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
        return self.rn.predict(x)

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

        layer = []
        for i in np.linspace(50,150,3,endpoint=True, dtype=np.int16):
            layer.append((i,))

        for i in np.linspace(50,150,3,endpoint=True, dtype=np.int16):
            for j in np.linspace(50,150,3,endpoint=True, dtype=np.int16):
                layer.append((i,j))


        classifier = MLPClassifier()
        # 'alpha' : np.linspace(0.001, 5, 15, endpoint=True),
        # 'power_t' : np.linspace(0.1, 2, 10, endpoint=True),
        # 'momentum' : np.linspace(0.1, 0.99, 10, endpoint=True),
        # 'beta_1' : np.linspace(0.1, 0.99, 10, endpoint=True),
        # 'beta_2' : np.linspace(0.1, 0.99, 10, endpoint=True),
        # 'n_iter_no_change' : np.linspace(5, 20, 16, endpoint=True, dtype=np.int16),
        # 'max_fun' : np.linspace(10000, 50000, 9, endpoint=True, dtype=np.int16),
        # 'learning_rate' : ['constant','invscaling', 'adaptive'],
        # 'learning_rate_init' : np.linspace(0.001, 5, 15, endpoint=True),
        # 'max_iter' : np.linspace(200, 1500, 5, endpoint=True, dtype=np.int16)
        parameters = {
            'hidden_layer_sizes' : layer,
            'activation' : ['identity','logistic', 'tanh', 'relu'],
            'solver' : ['lbfgs','sgd', 'adam'],
            'max_iter' : [500],
        }
        self.rn = GridSearchCV(classifier, parameters)
        self.rn.fit(x_tab, y_tab)

        self.hidden_layer_sizes = self.rn.best_estimator_.get_params()["hidden_layer_sizes"]
        self.activation = self.rn.best_estimator_.get_params()["activation"]
        self.solver = self.rn.best_estimator_.get_params()["solver"]
        self.alpha = self.rn.best_estimator_.get_params()["alpha"]
        self.learning_rate = self.rn.best_estimator_.get_params()["learning_rate"]
        self.learning_rate_init = self.rn.best_estimator_.get_params()["learning_rate_init"]
        self.power_t = self.rn.best_estimator_.get_params()["power_t"]
        self.max_iter = self.rn.best_estimator_.get_params()["max_iter"]
        self.momentum = self.rn.best_estimator_.get_params()["momentum"]
        self.beta_1 = self.rn.best_estimator_.get_params()["beta_1"]
        self.beta_2 = self.rn.best_estimator_.get_params()["beta_2"]
        self.n_iter_no_change = self.rn.best_estimator_.get_params()["n_iter_no_change"]
        self.max_fun = self.rn.best_estimator_.get_params()["max_fun"]