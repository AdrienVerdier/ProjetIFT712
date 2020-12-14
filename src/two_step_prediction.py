# -*- coding: utf-8 -*-

#####
# Félix Gaucher (gauf2611)
# Adrien Verdier (vera2704)
###

import numpy as np
import sys
import src.models.Plus_Proche_Voisin as PPV
import src.models.Analyse_discriminante_lineaire as ADL
import src.models.random_forest as RFC
import src.models.noyau as SVC
import src.models.perceptron as PCT
import src.models.reseau_neurone as RN

class TwoStepPrediction:

    def __init__(self):
        self.rechercheCat = [
            PPV.PlusProcheVoisin(),
            ADL.AnalyseDiscriminanteLineaire(),
            RFC.RandomForest(),
            SVC.Noyau(),
            PCT.PerceptronModele(),
            RN.ReseauNeurone()
        ]
        self.rechercheEspece = []
        for i in range (0,34):
            if i in [0,2,3,4,5,7,10,11,13,17,22,23,26,27]:
                self.rechercheEspece.append([
                    PPV.PlusProcheVoisin(),
                    ADL.AnalyseDiscriminanteLineaire(),
                    RFC.RandomForest(),
                    SVC.Noyau(),
                    PCT.PerceptronModele(),
                    RN.ReseauNeurone()
                ])
            else :
                self.rechercheEspece.append([])
    
    def entrainementModele(self, x_train, y_train, y_sous_categorie):
        """ Cette méthode va venir entrainer nos 2 modèles qui vont
        nous permettre de prédire les classes de nos feuilles
        La variable x_train contient
        les entrées (une matrice numpy avec tous nos éléments d'entrainement)
        et des cibles t_train (un tableau 1D Numpy qui contient les cibles des
        éléments d'entrainemnet).
        """

        self.entrainementRechercheCategorie(x_train, y_sous_categorie)
        self.entrainementRechercheSousCategorie(x_train, y_train, y_sous_categorie)

    def entrainementRechercheCategorie(self, x_train, y_sous_categorie):
        """
        Cette méthode va venir entrainer nos 6 modèles afin de reconnaitre
        la categorie auxquels appartiens chacun de nos ensemble 
        d'entrainement
        """

        for classifier in self.rechercheCat:
            classifier.validation_croisee(x_train, y_sous_categorie)
            #classifier.entrainement(x_train, y_sous_categorie)

    def entrainementRechercheSousCategorie(self, x_train, y_train, y_sous_categorie):
        """
        Cette méthode va venir entrainer nos modèles pour reconnaitre
        a quelle espèce appartient une données sachant qu'on connait 
        déjà la catégorie à laquelle elle appartient
        """

        for i in range (0,34):
            if i in [0,2,3,4,5,7,10,11,13,17,22,23,26,27]:
                x_t, y_t = self.recupElementCategorie(x_train, y_train, y_sous_categorie, i)

                for classifier in self.rechercheEspece[i]:
                    classifier.validation_croisee(x_t, y_t)
                    #classifier.entrainement(x_t, y_t)

    def predict(self, x):
        """
        Cette méthode permet de prédire l'espèce auquel
        appartiennent les données x
        """
        
        categorie = []
        for classifier in self.rechercheCat:
            categorie.append(np.array(classifier.prediction(x)))

        result = []
        for t in range (0,6) :
            inter = []
            for y in range(0, 6):
                inter.append(np.ones(len(x)))
            result.append(np.array(inter))
        result = np.array(result)

        for k in range(0, 6):
            for i in range (0,34):
                if i in [0,2,3,4,5,7,10,11,13,17,22,23,26,27]:
                    x_pred = []
                    id = []

                    for j in range (0, len(x)):
                        if categorie[k][j] == i :
                            x_pred.append(x[j])
                            id.append(j)

                    inter = []
                    for classifier in self.rechercheEspece[i]:
                        if len(x_pred) > 0:
                            inter.append(np.array(classifier.prediction(np.array(x_pred))))

                    for w in range(0, 6):
                        for n in range (0, len(id)):
                            result[k][w][id[n]] = inter[w][n]
                        
                elif i == 1 :
                    x_pred = []
                    id = []

                    for j in range(0, len(x)):
                        if categorie[k][j] == i :
                            x_pred.append(x[j])
                            id.append(j)

                    for w in range(0, 6) :
                        for n in range (0, len(id)):
                            result[k][w][id[n]] = 1
                elif i == 6 :
                    x_pred = []
                    id = []

                    for j in range(0, len(x)):
                        if categorie[k][j] == i :
                            x_pred.append(x[j])
                            id.append(j)

                    for w in range(0, 6) :
                        for n in range (0, len(id)):
                            result[k][w][id[n]] = 10
                elif i == 8 :
                    x_pred = []
                    id = []

                    for j in range(0, len(x)):
                        if categorie[k][j] == i :
                            x_pred.append(x[j])
                            id.append(j)

                    for w in range(0, 6) :
                        for n in range (0, len(id)):
                            result[k][w][id[n]] = 15
                elif i == 9 :
                    x_pred = []
                    id = []

                    for j in range(0, len(x)):
                        if categorie[k][j] == i :
                            x_pred.append(x[j])
                            id.append(j)

                    for w in range(0, 6) :
                        for n in range (0, len(id)):
                            result[k][w][id[n]] = 16
                elif i == 12 :
                    x_pred = []
                    id = []

                    for j in range(0, len(x)):
                        if categorie[k][j] == i :
                            x_pred.append(x[j])
                            id.append(j)

                    for w in range(0, 6) :
                        for n in range (0, len(id)):
                            result[k][w][id[n]] = 23
                elif i == 14 :
                    x_pred = []
                    id = []

                    for j in range(0, len(x)):
                        if categorie[k][j] == i :
                            x_pred.append(x[j])
                            id.append(j)

                    for w in range(0, 6) :
                        for n in range (0, len(id)):
                            result[k][w][id[n]] = 27
                elif i == 15 :
                    x_pred = []
                    id = []

                    for j in range(0, len(x)):
                        if categorie[k][j] == i :
                            x_pred.append(x[j])
                            id.append(j)

                    for w in range(0, 6) :
                        for n in range (0, len(id)):
                            result[k][w][id[n]] = 28
                elif i == 16 :
                    x_pred = []
                    id = []

                    for j in range(0, len(x)):
                        if categorie[k][j] == i :
                            x_pred.append(x[j])
                            id.append(j)

                    for w in range(0, 6) :
                        for n in range (0, len(id)):
                            result[k][w][id[n]] = 29
                elif i == 18 :
                    x_pred = []
                    id = []

                    for j in range(0, len(x)):
                        if categorie[k][j] == i :
                            x_pred.append(x[j])
                            id.append(j)

                    for w in range(0, 6) :
                        for n in range (0, len(id)):
                            result[k][w][id[n]] = 32
                elif i == 19 :
                    x_pred = []
                    id = []

                    for j in range(0, len(x)):
                        if categorie[k][j] == i :
                            x_pred.append(x[j])
                            id.append(j)

                    for w in range(0, 6) :
                        for n in range (0, len(id)):
                            result[k][w][id[n]] = 33
                elif i == 20 :
                    x_pred = []
                    id = []

                    for j in range(0, len(x)):
                        if categorie[k][j] == i :
                            x_pred.append(x[j])
                            id.append(j)

                    for w in range(0, 6) :
                        for n in range (0, len(id)):
                            result[k][w][id[n]] = 34
                elif i == 21 :
                    x_pred = []
                    id = []

                    for j in range(0, len(x)):
                        if categorie[k][j] == i :
                            x_pred.append(x[j])
                            id.append(j)

                    for w in range(0, 6) :
                        for n in range (0, len(id)):
                            result[k][w][id[n]] = 39
                elif i == 24 :
                    x_pred = []
                    id = []

                    for j in range(0, len(x)):
                        if categorie[k][j] == i :
                            x_pred.append(x[j])
                            id.append(j)

                    for w in range(0, 6) :
                        for n in range (0, len(id)):
                            result[k][w][id[n]] = 51
                elif i == 25 :
                    x_pred = []
                    id = []

                    for j in range(0, len(x)):
                        if categorie[k][j] == i :
                            x_pred.append(x[j])
                            id.append(j)

                    for w in range(0, 6) :
                        for n in range (0, len(id)):
                            result[k][w][id[n]] = 52
                elif i == 28 :
                    x_pred = []
                    id = []

                    for j in range(0, len(x)):
                        if categorie[k][j] == i :
                            x_pred.append(x[j])
                            id.append(j)

                    for w in range(0, 6) :
                        for n in range (0, len(id)):
                            result[k][w][id[n]] = 53
                elif i == 29 :
                    x_pred = []
                    id = []

                    for j in range(0, len(x)):
                        if categorie[k][j] == i :
                            x_pred.append(x[j])
                            id.append(j)

                    for w in range(0, 6) :
                        for n in range (0, len(id)):
                            result[k][w][id[n]] = 84
                elif i == 30 :
                    x_pred = []
                    id = []

                    for j in range(0, len(x)):
                        if categorie[k][j] == i :
                            x_pred.append(x[j])
                            id.append(j)

                    for w in range(0, 6) :
                        for n in range (0, len(id)):
                            result[k][w][id[n]] = 88
                elif i == 31 :
                    x_pred = []
                    id = []

                    for j in range(0, len(x)):
                        if categorie[k][j] == i :
                            x_pred.append(x[j])
                            id.append(j)

                    for w in range(0, 6) :
                        for n in range (0, len(id)):
                            result[k][w][id[n]] = 93
                elif i == 32 :
                    x_pred = []
                    id = []

                    for j in range(0, len(x)):
                        if categorie[k][j] == i :
                            x_pred.append(x[j])
                            id.append(j)

                    for w in range(0, 6) :
                        for n in range (0, len(id)):
                            result[k][w][id[n]] = 94
                elif i == 33 : 
                    x_pred = []
                    id = []

                    for j in range(0, len(x)):
                        if categorie[k][j] == i :
                            x_pred.append(x[j])
                            id.append(j)

                    for w in range(0, 6) :
                        for n in range (0, len(id)):
                            result[k][w][id[n]] = 95

        return result

        

    def recupElementCategorie(self, x_train, y_train, y_sous_categorie, categorie):
        """
        Récupère tous les éléments qui appartiennent à une catégorie
        """
        x_t = []
        y_t = []

        for i in range (0, len(x_train)):
            if y_sous_categorie[i] == categorie :
                x_t.append(x_train[i])
                y_t.append(y_train[i])

        return np.array(x_t), np.array(y_t)

