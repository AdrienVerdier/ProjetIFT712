# -*- coding: utf-8 -*-

#####
# Vos Noms (Vos Matricules) .~= À MODIFIER =~.
###

import numpy as np
import sys
import random
from sklearn.ensemble import RandomForestClassifier

class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='auto', max_leaf_nodes=None):
        """
        Classe effectuant de la classification de nos données dans les classes de résultats à l'aide de la méthode
        Random Forest (RandomForestClassifier)

        n_estimators: Le nombre d'arbre de notre forêt
        max_depth: La profondeur maximal d'un arbre
        min_samples_split: Le nombre minimum d'exemple pour découper un noeud
        min_samples_leaf: Le nombre minimum d'exemple qu'il faut pour être un noeud
        max_features: le nombre de features à regarder pour valider le meilleur découpage
        max_leaf_nodes: Le maximum de noeud par arbre
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.rfc = None

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
        Cette fonction trouve les meilleurs hyperaparamètres ``self.n_estimators``,
        ``self.max_depth``, ``self.min_samples_split``, ``self.min_samples_leaf``,
        ``self.max_features``, ``self.max_leaf_nodes``
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

        best_n_estimators = 0
        best_max_depth = 0
        best_min_samples_split = 2
        best_min_samples_leaf = 1
        best_max_features = ''
        features = ['sqrt','log2']
        best_max_leaf_nodes = 0

        best_error = 100000000

        for n_estimators_i in range (80,200,10):
            print("="*30)
            self.n_estimators = n_estimators_i
            print("n_estimators : " + str(n_estimators_i))
            print(" ")

            #for max_depth_i in range (1,50,5):
                #self.max_depth = max_depth_i

            for min_samples_leaf_i in range (1,3):
                self.min_samples_leaf = min_samples_leaf_i
                
                for min_samples_split_i in range (min_samples_leaf_i*2,(min_samples_leaf_i*2)+3):
                    self.min_samples_split = min_samples_split_i
                    
                    for max_features_i in features:
                        self.max_features = max_features_i

                        #for max_leaf_nodes_i in range (2,50):
		                #self.max_leaf_nodes = max_leaf_nodes_i

                        error = self.erreur_avec_parametre(k_samples, x_subsamples, y_subsamples)

                        if error <= best_error :
                            best_n_estimators = n_estimators_i
                            #best_max_leaf_nodes = max_leaf_nodes_i
                            best_max_features = max_features_i
                            best_min_samples_split = min_samples_split_i
                            best_min_samples_leaf = min_samples_leaf_i
                            #best_max_depth = max_depth_i
                            best_error = error

        # On met en place les meilleurs paramètres
        self.n_estimators = best_n_estimators
        #self.max_leaf_nodes = best_max_leaf_nodes
        self.max_features = best_max_features
        self.min_samples_split = best_min_samples_split
        self.min_samples_leaf = best_min_samples_leaf
        #self.max_depth = best_max_depth

        #print("meilleurs paramètres : " + str(best_n_estimators) + " " + str(best_max_depth) + " " + str(best_min_samples_leaf) + " " + str(best_min_samples_split) + " " + str(best_max_leaf_nodes) + " " + best_max_features)
        print("meilleurs paramètres : " + str(best_n_estimators)  + " " + str(best_min_samples_leaf) + " " +  str(best_max_leaf_nodes) + " " + best_max_features+ " " + str(best_min_samples_split))

	    # Dernière entrainement du modèle avec les meilleurs paramètres
        self.entrainement(x_tab, y_tab)

    def sauvegarde_modele() :
    	"""
    	Je ne sais pas trop comment la faire pour cette méthode là car
    	il faudrait pouvoir enregistrer notre modèle comme on le veut
    	afin de pouvoir le réutiliser juste pour une prédiction
    	"""
