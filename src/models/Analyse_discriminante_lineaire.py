# -*- coding: utf-8 -*-

#####
# Félix Gaucher (gauf2611)
# Adrien Verdier (vera2704)
###

import numpy as np
import sys
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
import random

class AnalyseDiscriminanteLineaire:
    def __init__(self, solver='svd', lda=None):
        """
        Classe effectuant de la classification de nos données dans les classes de résultats à l'aide de la méthode
        de l'analyse discriminante linéaire (LinearDiscriminantAnalysis)

        solver: le solver a utiliser pour notre modèle
        lda : Un modèle entrainé d'analyse discriminante linéaire
        """
        self.solver = solver
        if solver == 'lsqr' or solver == 'eigen':
            self.shrinkage = 'auto'
        else :
            self.shrinkage = None
        self.lda = lda

    def entrainement(self, x_train, y_train):
        """
        Entraîne une méthode d'apprentissage du type Analyse discriminante
        linéaire (LinearDiscriminantAnalysis). La variable x_train contient
        les entrées (une matrice numpy avec tous nos éléments d'entrainement)
        et des cibles t_train (un tableau 1D Numpy qui contient les cibles des
        éléments d'entrainemnet).
        """
        if self.solver != 'svd':
            lda = LinearDiscriminantAnalysis(solver = self.solver, shrinkage = self.shrinkage)
        else :
            lda = LinearDiscriminantAnalysis(solver = self.solver)
        lda.fit(x_train, y_train)
        self.lda = lda

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
        return self.lda.predict(x)

    def erreur(self, y, prediction):
        """
        Retourne la différence au carré entre
        la cible ``y`` et la prédiction ``prediction``.
        """
        return np.power(prediction - y, 2)

    def validation_croisee(self, x_tab, y_tab):
        """
        Cette fonction trouve le meilleur solveur ``self.solver``,
        avec une validation croisée de type "k-fold" où k=5 avec les
        données contenues dans x_tab et y_tab. Une fois le meilleur
        hyperparamètre trouvé, le modèle est entraîné une dernière fois.
        """

        classifier = LinearDiscriminantAnalysis()
        parameters = {'solver':('svd', 'lsqr', 'eigen')}
        self.lda = GridSearchCV(classifier, parameters)
        self.lda.fit(x_tab, y_tab)

        self.solver = self.lda.best_estimator_.get_params()["solver"]

    def sauvegarde_modele() :
        """
        Je ne sais pas trop comment la faire pour cette méthode là car
        il faudrait pouvoir enregistrer notre modèle comme on le veut
        afin de pouvoir le réutiliser juste pour une prédiction
        """

        # Voila comment on accède aux données si on veut les enregistrer
        print("solver : " + self.solver)