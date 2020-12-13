# -*- coding: utf-8 -*-

#####
# Félix Gaucher (gauf2611)
# Adrien Verdier (vera2704)
###

import numpy as np
import sys
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='auto', max_leaf_nodes=None, rfc=None):
        """
        Classe effectuant de la classification de nos données dans les classes de résultats à l'aide de la méthode
        Random Forest (RandomForestClassifier)

        n_estimators: Le nombre d'arbre de notre forêt
        max_depth: La profondeur maximal d'un arbre
        min_samples_split: Le nombre minimum d'exemple pour découper un noeud
        min_samples_leaf: Le nombre minimum d'exemple qu'il faut pour être un noeud
        max_features: le nombre de features à regarder pour valider le meilleur découpage
        max_leaf_nodes: Le maximum de noeud par arbre
        rfc : Un modèle random forest entrainé
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.rfc = rfc

    def entrainement(self, x_train, y_train):
        """
        Entraîne une méthode d'apprentissage du type Random Forest
        (RandomForestClassifier). La variable x_train contient
        les entrées (une matrice numpy avec tous nos éléments d'entrainement)
        et des cibles t_train (un tableau 1D Numpy qui contient les cibles des
        éléments d'entrainemnet).
        """
        rfc = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf, max_features=self.max_features, max_leaf_nodes=self.max_leaf_nodes)
        rfc.fit(x_train, y_train)
        self.rfc = rfc

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
        return self.rfc.predict(x)

    def erreur(self, y, prediction):
        """
        Retourne la différence au carré entre
        la cible ``y`` et la prédiction ``prediction``.
        """
        return np.power(prediction - y, 2)

    def validation_croisee(self, x_tab, y_tab):
        """
        Cette fonction trouve les meilleurs hyperaparamètres ``self.n_estimators``,
        ``self.max_depth``, ``self.min_samples_leaf``,
        ``self.max_features``, ``self.max_leaf_nodes``
        avec une validation croisée de type "k-fold" où k=5 avec les
        données contenues dans x_tab et y_tab. Une fois le meilleur
        hyperparamètre trouvé, le modèle est entraîné une dernière fois.
        """

        classifier = RandomForestClassifier()
        # , 'max_depth':[10,30,50], 'min_samples_leaf':np.linspace(1, 3, 3, endpoint=True, dtype=np.int16), 'max_leaf_nodes':[10,30,50]
        parameters = {'n_estimators':np.linspace(80, 200, 13, endpoint=True, dtype=np.int16), 'max_features':('sqrt', 'log2')}
        self.rfc = GridSearchCV(classifier, parameters)
        self.rfc.fit(x_tab, y_tab)

        self.n_estimators = self.rfc.best_estimator_.get_params()["n_estimators"]
        self.max_depth = self.rfc.best_estimator_.get_params()["max_depth"]
        self.min_samples_leaf = self.rfc.best_estimator_.get_params()["min_samples_leaf"]
        self.max_features = self.rfc.best_estimator_.get_params()["max_features"]
        self.max_leaf_nodes = self.rfc.best_estimator_.get_params()["max_leaf_nodes"]