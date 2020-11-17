# -*- coding: utf-8 -*-

#####
# Vos Noms (Vos Matricules) .~= À MODIFIER =~.
###

import numpy as np
import sys
from sklearn.neighbors import KNeighborsClassifier
import random

class PlusProcheVoisin:
    def __init__(self, nb_voisin=5, weights='uniform', algorithm='auto', ppv=None):
        """
        Classe effectuant de la classification de nos données dans les classes de résultats à l'aide de la méthode
         des K plus proches voisins (KNeighborsClassifier)

        nb_voisin: le nombre de voisin à prendre comme paramètre
        weights: Si les poids sont uniformes ou s'ils dépendent de la distance entre les points
        algorithm: algorithm utilisé par le noyau (auto, ball_tree, kd_tree, brute)
        ppv: un modèle entrainer de plus proches voisins
        """
        self.nb_voisin = nb_voisin
        self.weights = weights
        self.algorithm = algorithm
        self.ppv = None

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

    def erreur_avec_parametre(self, k_samples, x_subsamples, y_subsamples):
        """
        Cette méthode va venir entrainer notre modèles sur différents batch de
        données avec les mêmes paramètres et effectuer des prédictions sur des
        batch de tests. Elle va ensuite calculer la somme de nos erreurs
        pour les renvoyer. Cette méthode va nous être utile pour notre
        validation croisée et notre recherche d'hyper-paramètres

        k_samples représente le nomre de sample de données qu'on a
        x_subsamples représente nos différents ensemble de données (Liste de matrice numpy)
        y_subsamples représente nos différents ensemble de cible (Liste de liste numpy)

        Cette méthode va nous renvoyer la somme de toutes les erreurs de nos
        modèles qui ont été entrainé sur des batch de données différents avec
        les mêmes paramètres
        """
        error = 0.0

        for split_indice in range(0, k_samples):
            # We train with all the data except the slit_num sample
            x_training_data = []
            y_training_data = []
            for k in range(0, k_samples):
                if k != split_indice:
                    x_training_data.extend(x_subsamples[k])
                    y_training_data.extend(y_subsamples[k])

            self.entrainement(np.array(x_training_data), np.array(y_training_data))
            error += np.array([self.erreur(t_n, p_n) for t_n, p_n in zip(y_subsamples[split_indice], np.array(
                self.prediction(x_subsamples[split_indice])))]).sum()

        return error

    def validation_croisee(self, x_tab, y_tab):
        """
        Cette fonction trouve les meilleurs hyperparametres ``self.nb_voisin``,
        ``self.weights`` et ``self.algorithm`` avec une validation croisée de
        type "k-fold" où k=5 avec les données contenues dans x_tab et y_tab.
        Une fois les meilleurs hyperparamètres trouvés, le modèle est entraîné
        une dernière fois.
        """
        k_samples = 5

        # On mélange les données
        shuffling_temp = list(zip(x_tab, y_tab))
        random.shuffle(shuffling_temp)
        shuffled_x, shuffled_y = zip(*shuffling_temp)

        # Si k est plus grand que le nombre de points, on le réduit au nombre de points
        if len(shuffled_x) < k_samples:
            k_samples = len(x_tab)

        # We start by separating X and t into k sub-samples
        x_subsamples = np.array_split(shuffled_x, k_samples)
        y_subsamples = np.array_split(shuffled_y, k_samples)

        best_nb_voisin = 0
        best_weights = ''
        weights = ['uniform', 'distance']
        best_algorithm = ''
        algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']

        best_error = 100000000

        for algorithm_i in algorithm:
            print("="*30)
            self.algorithm = algorithm_i
            print("Algorithme : " + algorithm_i)
            print(" ")

            for weights_i in weights:
                self.weights = weights_i
                print("Weights :" + weights_i)

                for nb_voisin_i in range (1,16):
                    self.nb_voisin = nb_voisin_i

                    error = self.erreur_avec_parametre(k_samples, x_subsamples,     y_subsamples)
                    print("nombre voisin : " + str(nb_voisin_i))
                    print("error : " + str(error))
                    print (" ")

                    if error <= best_error and nb_voisin_i >= best_nb_voisin :
                        best_nb_voisin = nb_voisin_i
                        best_algorithm = algorithm_i
                        best_weights = weights_i
                        best_error = error

        # On met en place les meilleurs paramètres
        self.nb_voisin = best_nb_voisin
        self.algorithm = best_algorithm
        self.weights = best_weights

        print("meilleurs paramètres : " + str(best_nb_voisin) + " " + best_algorithm + " " + best_weights)

        # Dernière entrainement du modèle avec les meilleurs paramètres
        self.entrainement(x_tab, y_tab)

    def sauvegarde_modele() :
        """
        Je ne sais pas trop comment la faire pour cette méthode là car
        il faudrait pouvoir enregistrer notre modèle comme on le veut
        afin de pouvoir le réutiliser juste pour une prédiction
        """
