#####
# FÃ©lix Gaucher (gauf2611)
# Adrien Verdier (vera2704)
###

import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit

class DecoupeDonnees:
    def __init__(self, data_train, labels_train, labels_categ_train, pourcentage=0.2, mode=1):
        """
        data_train : nos donnÃ©es d'entrainement
        labels_train : les rÃ©sultats de nos donnÃ©es d'entrainement
        labels_categ_train : les catÃ©gories de nos donnÃ©es d'entrainement
        pourcentage : le pourcentage de nos donnÃ©es qu'on veut alouer aux test_size
        mode : mode de prÃ©paration de nos donnÃ©es qu'on veut effectuer
        """
        self.data_train = data_train
        self.labels_train = labels_train
        self.labels_categ_train = labels_categ_train
        self.pourcentage = pourcentage
        self.mode = mode

    def mise_en_forme_donnees(self):
        """
        Fonction qui va mettre en forme les donnÃ©es pour pouvoir entrainer nos modÃ¨les efficacement

        Returns:
            x_train [np.array]: La liste de nos donnÃ©es d'entrainement
            y_train [np.array]: La liste des rÃ©sultats attendu de nos donnÃ©es d'entrainement
            x_test [np.array]: La liste de nos donnÃ©es de test
            y_test [np.array]: La liste des rÃ©sultats attendu de nos donnÃ©es de test
            t_categ_train [np.array] : La liste des catÃ©gories de nos donnÃ©es d'entrainement
        """
        [x_train, y_train, x_test, y_test, t_categ_train] = self.decouper_donnees()

        return x_train, y_train, x_test, y_test, t_categ_train

    def decouper_donnees(self):
        """
        Cette mÃ©thode va dÃ©couper notre ensemble de donnÃ©es en donnÃ©es de test/entrainement

        Args:

        Returns:
            x_train [np.array]: matrice comportant nos donnÃ©es d'entrainement
            y_train [np.array]: matrice comportant les rÃ©sultats de nos donnÃ©es d'entrainement
            x_test [np.array]: matrice comportant nos donnÃ©es de tests
            y_test [np.array]: matrice comportant les rÃ©sultats de nos donnÃ©es de tests
            t_categ_train [np.array] : La liste des catÃ©gories de nos donnÃ©es d'entrainement
        """
        melange = StratifiedShuffleSplit(len(self.labels_train), test_size=self.pourcentage, random_state=17)

        for train_index, test_index in melange.split(self.data_train,self.labels_train):
            x_train, x_test = self.data_train[train_index], self.data_train[test_index]
            y_train, y_test = self.labels_train[train_index], self.labels_train[test_index]
            t_categ_train = self.labels_categ_train[train_index]

        return x_train, y_train, x_test, y_test, t_categ_train

    def get_scaled_data(self, x_train, x_test):
        """
        Cette mÃ©thode va transformer les donnÃ©es grÃ¢ce Ã  une mÃ©thode de prÃ©-traitement de scikit learn
        pour fournir des donnÃ©es dite scalÃ©
        
        Args:
            x_train (np.array): données d'entrainement devant être transformées
            x_test (np.array): données de test devant être transformées
            
        Returns:
            x_train_scaled (np.array): données d'entrainement transformées
            x_test_scaled (np.array): données de test transformées
        """
        x_train_scaled = preprocessing.scale(x_train)
        x_test_scaled = preprocessing.scale(x_test)

        return x_train_scaled, x_test_scaled

    def get_min_max_data(self, x_train, x_test):
        """
        Cette mÃ©thode va venir modifier nos donnÃ©es brute grÃ¢ce Ã  la mÃ©thode
        Min Max de prÃ©processing de scikit learn
        
        Args:
            x_train (np.array): données d'entrainement devant être transformées
            x_test (np.array): données de test devant être transformées
            
        Returns:
            x_train_minmax (np.array): données d'entrainement transformées
            x_test_minmax (np.array): données de test transformées
        """
        min_max_scaler = preprocessing.MinMaxScaler()
        x_train_minmax = min_max_scaler.fit_transform(x_train)
        x_test_minmax = min_max_scaler.fit_transform(x_test)

        return x_train_minmax, x_test_minmax

    def get_max_abs_data(self, x_train, x_test):
        """
        Cette mÃ©thode va venir modifier nos donnÃ©es brute grÃ¢ce Ã  la mÃ©thode 
        Max abs de prÃ©processing de scikit learn
        
        Args:
            x_train (np.array): données d'entrainement devant être transformées
            x_test (np.array): données de test devant être transformées
            
        Returns:
            x_train_maxabs (np.array): données d'entrainement transformées
            x_test_maxabs (np.array): données de test transformées
        """
        max_abs_scaler = preprocessing.MaxAbsScaler()
        x_train_maxabs = max_abs_scaler.fit_transform(x_train)
        x_test_maxabs = max_abs_scaler.fit_transform(x_test)

        return x_train_maxabs, x_test_maxabs

    def get_quantile_data(self, x_train, x_test):
        """
        Cette mÃ©thode va venir modifier nos donnÃ©es brute grÃ¢ce Ã  la mÃ©thode 
        quantile de prÃ©processing de scikit learn
        
        Args:
            x_train (np.array): données d'entrainement devant être transformées
            x_test (np.array): données de test devant être transformées
            
        Returns:
            x_train_trans (np.array): données d'entrainement transformées
            x_test_trans (np.array): données de test transformées
        """
        quantile_transformer = preprocessing.QuantileTransformer(random_state=17)
        x_train_trans = quantile_transformer.fit_transform(x_train)
        x_test_trans = quantile_transformer.fit_transform(x_test)

        return x_train_trans, x_test_trans

    def get_gaussian_data(self, x_train, x_test):
        """
        Cette mÃ©thode va venir modifier nos donnÃ©es brute grÃ¢ce Ã  la mÃ©thode 
        Gaussian de prÃ©processing de scikit learn
        
        Args:
            x_train (np.array): données d'entrainement devant être transformées
            x_test (np.array): données de test devant être transformées
            
        Returns:
            x_train_gauss (np.array): données d'entrainement transformées
            x_test_gauss (np.array): données de test transformées
        """
        quantile_transformer2 = preprocessing.QuantileTransformer(output_distribution='normal', random_state=17)
        x_train_gauss = quantile_transformer2.fit_transform(x_train)
        x_test_gauss = quantile_transformer2.fit_transform(x_test)

        return x_train_gauss, x_test_gauss

    def get_normalized_data(self, x_train, x_test):
        """
        Cette mÃ©thode va venir modifier nos donnÃ©es brute grÃ¢ce Ã  la mÃ©thode 
        normalize de prÃ©processing de scikit learn
        
        Args:
            x_train (np.array): données d'entrainement devant être transformées
            x_test (np.array): données de test devant être transformées
            
        Returns:
            x_train_normalized (np.array): données d'entrainement transformées
            x_test_normalized (np.array): données de test transformées
        """
        x_train_normalized = preprocessing.normalize(x_train, 'l2')
        x_test_normalized = preprocessing.normalize(x_test, 'l2')

        return x_train_normalized, x_test_normalized
