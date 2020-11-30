#####
# Félix Gaucher (gauf2611)
# Adrien Verdier (vera2704)
###

import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit

class DecoupeDonnees:
    def __init__(self, data_train, labels_train, labels_categ_train, pourcentage=0.2, mode=1):
        """
        data_train : nos données d'entrainement
        labels_train : les résultats de nos données d'entrainement
        labels_categ_train : les catégories de nos données d'entrainement
        pourcentage : le pourcentage de nos données qu'on veut alouer aux test_size
        mode : mode de préparation de nos données qu'on veut effectuer
        """
        self.data_train = data_train
        self.labels_train = labels_train
        self.labels_categ_train = labels_categ_train
        self.pourcentage = pourcentage
        self.mode = mode

    def mise_en_forme_donnees(self):
        """
        Fonction qui va mettre en forme les données pour pouvoir entrainer nos modèles efficacement

        Returns:
            x_train [np.array]: La liste de nos données d'entrainement
            y_train [np.array]: La liste des résultats attendu de nos données d'entrainement
            x_test [np.array]: La liste de nos données de test
            y_test [np.array]: La liste des résultats attendu de nos données de test
            t_categ_train [np.array] : La liste des catégories de nos données d'entrainement
        """
        [x_train, y_train, x_test, y_test, t_categ_train] = self.decouper_donnees()
        #self.enregistrer_donnees(x_train, y_train, x_test, y_test)

        #Ensuite on ajoutera nos méthodes qui vont venir modifier nos données, enfin dans cette méthode là, ça pourra être avant

        return x_train, y_train, x_test, y_test, t_categ_train

    def decouper_donnees(self):
        """
        Cette méthode va découper notre ensemble de données en données de test/entrainement

        Args:

        Returns:
            x_train [np.array]: matrice comportant nos données d'entrainement
            y_train [np.array]: matrice comportant les résultats de nos données d'entrainement
            x_test [np.array]: matrice comportant nos données de tests
            y_test [np.array]: matrice comportant les résultats de nos données de tests
            t_categ_train [np.array] : La liste des catégories de nos données d'entrainement
        """
        melange = StratifiedShuffleSplit(len(self.labels_train), test_size=self.pourcentage, random_state=17)

        for train_index, test_index in melange.split(self.data_train,self.labels_train):
            x_train, x_test = self.data_train[train_index], self.data_train[test_index]
            y_train, y_test = self.labels_train[train_index], self.labels_train[test_index]
            t_categ_train = self.labels_categ_train[train_index]

        return x_train, y_train, x_test, y_test, t_categ_train

    def get_scaled_data(self, x_train, x_text):
        """
        Cette méthode va transformer les données grâce à une méthode de pré-traitement de scikit learn
        pour fournir des données dite scalé
        """
        x_train_scaled = preprocessing.scale(x_train)
        x_test_scaled = preprocessing.scale(x_test)

        return x_train_scaled, x_test_scaled

    def get_min_max_data(self, x_train, x_test):
        """
        Cette méthode va venir modifier nos données brute grâce à la méthode
        Min Max de préprocessing de scikit learn
        """
        min_max_scaler = preprocessing.MinMaxScaler()
        x_train_minmax = min_max_scaler.fit_transform(x_train)
        x_test_minmax = min_max_scaler.fit_transform(x_test)

        return x_train_minmax, x_test_minmax

    def get_max_abs_data(self, x_train, x_test):
        """
        Cette méthode va venir modifier nos données brute grâce à la méthode 
        Max abs de préprocessing de scikit learn
        """
        max_abs_scaler = preprocessing.MaxAbsScaler()
        x_train_maxabs = max_abs_scaler.fit_transform(x_train)
        x_test_maxabs = max_abs_scaler.fit_transform(x_test)

        return x_train_maxabs, x_test_maxabs

    def get_quantile_data(self, x_train, x_test):
        """
        Cette méthode va venir modifier nos données brute grâce à la méthode 
        quantile de préprocessing de scikit learn
        """
        quantile_transformer = preprocessing.QuantileTransformer(random_state=17)
        x_train_trans = quantile_transformer.fit_transform(x_train)
        x_test_trans = quantile_transformer.fit_transform(x_test)

        return x_train_trans, x_test_trans

    def get_gaussian_data(self, x_train, x_test):
        """
        Cette méthode va venir modifier nos données brute grâce à la méthode 
        Gaussian de préprocessing de scikit learn
        """
        quantile_transformer2 = preprocessing.QuantileTransformer(output_distribution='normal', random_state=17)
        x_train_gauss = quantile_transformer2.fit_transform(x_train)
        x_test_gauss = quantile_transformer2.fit_transform(x_test)

        return x_train_gauss, x_test_gauss

    def get_normalized_data(self, x_train, x_test):
        """
        Cette méthode va venir modifier nos données brute grâce à la méthode 
        normalize de préprocessing de scikit learn
        """
        x_train_normalized = preprocessing.normalize(x_train, 'l2')
        x_test_normalized = preprocessing.normalize(x_test, 'l2')

        return x_train_normalized, x_test_normalized

    def enregistrer_donnees(self, x_train, y_train, x_test, y_test):
        """
        Cette méthode va venir enregistrer toutes nos données dans un fichier sur le git

        Args:
            x_train (numpy array): Liste de toutes nos données d'entrainement
            y_train (numpy array): Liste de toutes nos classes qui correspondent aux différentes données d'entrainement
            x_test (numpy array): Liste de toutes les données de tests
            y_test (numpy array): Liste de toutes nos classes qui correspondent aux différentes données d'entrainement
        """

        # Coder cette méthode pour enregistrer les données dans un fichier
