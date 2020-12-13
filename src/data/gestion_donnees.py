#####
# Félix Gaucher (gauf2611)
# Adrien Verdier (vera2704)
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
        t_train, t_categ_train = self.mise_en_forme_classes(classes, t_train)
        #self.enregistrer_donnees(x_train, t_train, x_test, classes)

        return x_train, t_train, x_test, classes, t_categ_train

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

        labels_train_preApprentissage = []
        for label in labels_train:
            categorie = label.split("_")

            labels_train_preApprentissage.append(self.findClasses(categorie[0])-1)

        return np.array(labels_train_modifie), np.array(labels_train_preApprentissage)

    def findClasses(self, categorie):
        """Cette méthode permet de renvoyer le numéro de la catégorie qui lui correspond

        Args:
            categorie (String): Catégorie de la feuille
        """
        if categorie == "Acer":
            return 1
        elif categorie == "Pterocarya":
            return 2
        elif categorie == "Quercus":
            return 3
        elif categorie == "Tilia":
            return 4
        elif categorie == "Magnolia":
            return 5
        elif categorie == "Salix":
            return 6
        elif categorie == "Zelkova":
            return 7
        elif categorie == "Betula":
            return 8
        elif categorie == "Fagus":
            return 9
        elif categorie == "Phildelphus":
            return 10
        elif categorie == "Populus":
            return 11
        elif categorie == "Alnus":
            return 12
        elif categorie == "Arundinaria":
            return 13
        elif categorie == "Cornus":
            return 14
        elif categorie == "Liriodendron":
            return 15
        elif categorie == "Cytisus":
            return 16
        elif categorie == "Rhododendron":
            return 17
        elif categorie == "Eucalyptus":
            return 18
        elif categorie == "Cercis":
            return 19
        elif categorie == "Cotinus":
            return 20
        elif categorie == "Celtis":
            return 21
        elif categorie == "Callicarpa":
            return 22
        elif categorie == "Prunus":
            return 23
        elif categorie == "Ilex":
            return 24
        elif categorie == "Ginkgo":
            return 25
        elif categorie == "Liquidambar":
            return 26
        elif categorie == "Lithocarpus":
            return 27
        elif categorie == "Viburnum":
            return 28
        elif categorie == "Crataegus":
            return 29
        elif categorie == "Morus":
            return 30
        elif categorie == "Olea":
            return 31
        elif categorie == "Castanea":
            return 32
        elif categorie == "Ulmus":
            return 33
        elif categorie == "Sorbus":
            return 34