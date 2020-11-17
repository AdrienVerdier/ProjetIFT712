#####
# VotreNom (VotreMatricule) .~= À MODIFIER =~.
###

import csv
import numpy as np

class GestionDonnees:
    def __init__(self, train_file, test_file):
        """
        train_file : chemin de notre fichier d'entrainement
        test_file : chemin de notre fichier de test
        """
        self.train_file = train_file
        self.test_file = test_file

    def recuperer_donnees(self):
        """
        Fonction qui récupère les données d'entrainement et de test dans les fichiers

        train_file : nom du fichier d'entrainement
        test_file : nom du fichier de test
        """
        x_train, t_train, classes = self.lire_fichier_entrainement(self.train_file)
        x_test = self.lire_fichier_test(self.test_file)
        t_train = self.mise_en_forme_classes(classes, t_train)
        #self.enregistrer_donnees(x_train, t_train, x_test, classes)

        return x_train, t_train, x_test, classes

    def lire_fichier_entrainement(self, train_file):
        """
        Fonction qui lit le fichier d'entrainement pour en extraire nos informations

        Args:
            train_file (String): Le nom de notre fichier

        Returns:
            x_train [np.array]: matrice contenant nos données d'entrainement
            t_train [np.array]: vecteur contenant les résultats attendu de nos données d'entrainement
            classes [np.array]: nom de nos classes possibles
        """
        file = open(train_file, "r")

        try:
            reader = csv.reader(file)

            x_train = []
            t_train = []
            classes = []
            ligne = 0

            for row in reader:
                if(ligne != 0):
                    feuille = []

                    t_train.append(row[1])
                    if row[1] not in classes:
                        classes.append(row[1])

                    for i in range (2,len(row)):
                        feuille.append(float(row[i]))

                    x_train.append(feuille)

                ligne += 1
        finally:
            file.close()

        return np.array(x_train), np.array(t_train), np.array(classes)

    def lire_fichier_test(self, test_file):
        """
        Fonction qui lit le fichier de test pour en extraire nos informations

        Args:
            test_file (String): Le nom de notre fichier à ouvrir

        Returns:
            x_test [np.array]: matrice contenant nos données de test
        """
        file = open(test_file, "r")

        try:
            reader = csv.reader(file)

            x_test = []
            ligne = 0

            for row in reader:
                if(ligne != 0):
                    feuille = []


                    for i in range (1, len(row)):
                        feuille.append(float(row[i]))

                    x_test.append(feuille)

            ligne += 1

        finally:
            file.close()

        return np.array(x_test)

    def mise_en_forme_classes(self, classes, labels_train):
        """
        Fonction qui va affecter un chiffre à une classe qui correspond à son emplacement dans le tableau des classes

        Args:
            classes (numpy array): Liste de tous nos noms de classes
            labels_train (numpy array): Liste de toutes nos classes qui correspondent aux différentes données d'entrainement

        Returns:
            labels_train_modifie[np.array]: Liste de l'indice de nos classes qui correspondent aux différentes données d'entrainement
        """
        labels_train_modifie = []

        for i in range (0,len(labels_train)):
                labels_train_modifie.append(int(np.where(classes==labels_train[i])[0][0]))

        return np.array(labels_train_modifie)

    def enregistrer_donnees(self, data_train, labels_train, data_test, classes):
        """
        Cette méthode va venir enregistrer toutes nos données dans un fichier sur le git

        Args:
            data_train (numpy array): Liste de toutes nos données d'entrainement
            labels_train (numpy array): Liste de toutes nos classes qui correspondent aux différentes données d'entrainement
            data_test (numpy array): Liste de toutes les données de tests
            classes (numpy array): Liste de tous nos noms de classes
        """

        # Coder cette méthode pour enregistrer les données dans un fichier
