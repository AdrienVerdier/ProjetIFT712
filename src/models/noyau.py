# -*- coding: utf-8 -*-

#####
# Vos Noms (Vos Matricules) .~= À MODIFIER =~.
###

import numpy as np
import sys
from sklearn.svm import SVC, LinearSVC
import random

class Noyau:
    def __init__(self, kernel="rbf", C=0.025, probability=True):
        """
        Classe effectuant de la classification de nos données dans les classes de résultats à l'aide de la méthode
        du noyau (SVC)

        kernel: le noyau a utiliser
        C: Le paramètre de régularisation
        probability: Utilisation les estimations probabilistes
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
        Cette fonction trouve les meilleurs hyperparamètres ``self.kernel``,
        ``self.C``, ``self.probability``
        avec une validation croisée de type "k-fold" où k=5 avec les
        données contenues dans x_tab et y_tab. Une fois le meilleur
        hyperparamètre trouvé, le modèle est entraîné une dernière fois.
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

        best_C = 0
        best_kernel = ''
        kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
        best_probability = True

        best_error = 100000000

        for kernel_i in kernels:
            print("="*30)
            self.kernel = kernel_i
            print("Algorithme : " + kernel_i)
            print(" ")

            for probability_i in [True, False]:
                self.probability = probability_i
                print("Probability :" + probability_i)

                for C_i in np.linspace(0.000000001, 2, 25):
                    self.C = C_i

                    error = self.erreur_avec_parametre(k_samples, x_subsamples, y_subsamples)
                    print("C : " + str(C_i))
                    print("error : " + str(error))
                    print (" ")

                    if error <= best_error :
                        best_C = C_i
                        best_kernel = kernel_i
                        best_probability = probability_i
                        best_error = error

        # On met en place les meilleurs paramètres
        self.C = best_C
        self.kernel = best_kernel
        self.probability = best_probability

        print("meilleurs paramètres : " + str(best_C) + " " + best_kernel + " " + best_probability)

        # Dernière entrainement du modèle avec les meilleurs paramètres
        self.entrainement(x_tab, y_tab)

    def sauvegarde_modele() :
    	"""
    	Je ne sais pas trop comment la faire pour cette méthode là car
    	il faudrait pouvoir enregistrer notre modèle comme on le veut
    	afin de pouvoir le réutiliser juste pour une prédiction
    	"""
