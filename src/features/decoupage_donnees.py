#####
# VotreNom (VotreMatricule) .~= À MODIFIER =~.
###

#import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit

class DecoupeDonnees:
    def __init__(self, data_train, labels_train, pourcentage=0.2, mode=1):
        """
        data_train : nos données d'entrainement
        labels_train : les résultats de nos données d'entrainement
        pourcentage : le pourcentage de nos données qu'on veut alouer aux test_size
        mode : mode de préparation de nos données qu'on veut effectuer
        """
        self.data_train = data_train
        self.labels_train = labels_train
        self.pourcentage = pourcentage
        self.mode = mode

    @staticmethod
    def mise_en_forme_donnees(self):
        """
        Fonction qui va mettre en forme les données pour pouvoir entrainer nos modèles efficacement

        Returns:
            x_train [np.array]: La liste de nos données d'entrainement
            y_train [np.array]: La liste des résultats attendu de nos données d'entrainement
            x_test [np.array]: La liste de nos données de test
            y_test [np.array]: La liste des résultats attendu de nos données de test
        """
        [x_train, y_train, x_test, y_test] = self.decouper_donnees()
        #self.enregistrer_donnees(x_train, y_train, x_test, y_test)

        #Ensuite on ajoutera nos méthodes qui vont venir modifier nos données, enfin dans cette méthode là, ça pourra être avant

        return x_train, y_train, x_test, y_test

    def decouper_donnees(self):
        """
        Cette méthode va découper notre ensemble de données en données de test/entrainement

        Args:

        Returns:
            x_train [np.array]: matrice comportant nos données d'entrainement
            y_train [np.array]: matrice comportant les résultats de nos données d'entrainement
            x_test [np.array]: matrice comportant nos données de tests
            y_test [np.array]: matrice comportant les résultats de nos données de tests
        """
        melange = StratifiedShuffleSplit(len(self.labels_train), test_size=self.pourcentage, random_state=17)

        for train_index, test_index in melange.split(self.data_train,self.labels_train):
            x_train, x_test = self.data_train[train_index], self.data_train[test_index]
            y_train, y_test = self.labels_train[train_index], self.labels_train[test_index]

        return x_train, y_train, x_test, y_test

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