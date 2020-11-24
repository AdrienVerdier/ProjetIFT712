#####
# VotreNom (VotreMatricule) .~= � MODIFIER =~.
###

#import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit

class DecoupeDonnees:
    def __init__(self, data_train, labels_train, pourcentage=0.2, mode=1):
        """
        data_train : nos donn�es d'entrainement
        labels_train : les r�sultats de nos donn�es d'entrainement
        pourcentage : le pourcentage de nos donn�es qu'on veut alouer aux test_size
        mode : mode de pr�paration de nos donn�es qu'on veut effectuer
        """
        self.data_train = data_train
        self.labels_train = labels_train
        self.pourcentage = pourcentage
        self.mode = mode

    @staticmethod
    def mise_en_forme_donnees(self):
        """
        Fonction qui va mettre en forme les donn�es pour pouvoir entrainer nos mod�les efficacement

        Returns:
            x_train [np.array]: La liste de nos donn�es d'entrainement
            y_train [np.array]: La liste des r�sultats attendu de nos donn�es d'entrainement
            x_test [np.array]: La liste de nos donn�es de test
            y_test [np.array]: La liste des r�sultats attendu de nos donn�es de test
        """
        [x_train, y_train, x_test, y_test] = self.decouper_donnees()
        #self.enregistrer_donnees(x_train, y_train, x_test, y_test)

        #Ensuite on ajoutera nos m�thodes qui vont venir modifier nos donn�es, enfin dans cette m�thode l�, �a pourra �tre avant

        return x_train, y_train, x_test, y_test

    def decouper_donnees(self):
        """
        Cette m�thode va d�couper notre ensemble de donn�es en donn�es de test/entrainement

        Args:

        Returns:
            x_train [np.array]: matrice comportant nos donn�es d'entrainement
            y_train [np.array]: matrice comportant les r�sultats de nos donn�es d'entrainement
            x_test [np.array]: matrice comportant nos donn�es de tests
            y_test [np.array]: matrice comportant les r�sultats de nos donn�es de tests
        """
        melange = StratifiedShuffleSplit(len(self.labels_train), test_size=self.pourcentage, random_state=17)

        for train_index, test_index in melange.split(self.data_train,self.labels_train):
            x_train, x_test = self.data_train[train_index], self.data_train[test_index]
            y_train, y_test = self.labels_train[train_index], self.labels_train[test_index]

        return x_train, y_train, x_test, y_test

    def enregistrer_donnees(self, x_train, y_train, x_test, y_test):
        """
        Cette m�thode va venir enregistrer toutes nos donn�es dans un fichier sur le git

        Args:
            x_train (numpy array): Liste de toutes nos donn�es d'entrainement
            y_train (numpy array): Liste de toutes nos classes qui correspondent aux diff�rentes donn�es d'entrainement
            x_test (numpy array): Liste de toutes les donn�es de tests
            y_test (numpy array): Liste de toutes nos classes qui correspondent aux diff�rentes donn�es d'entrainement
        """

        # Coder cette m�thode pour enregistrer les donn�es dans un fichier