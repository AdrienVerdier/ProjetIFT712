# -*- coding: utf-8 -*-

#####
# Vos Noms (Vos Matricules) .~= À MODIFIER =~.
###

import numpy as np
import sys
import gestion_donnees as gd
import decoupage_donnees as dd

#Provisoir :
import pandas as pd

#Ajouter une méthodes pour le sur/sous-entrainement

def main():
    # Pour l'exécution du programme, on peut faire une fast exécution, une exécution complète , juste une méthode en particulier fast, une exécution une méthode en particulier complète

    # On peut ajouter la récupération des arguments comme dans les TPs
    train_file = 'train.csv'
    test_file = 'test.csv'

    # On récupère les données d'entrainement et de test (même si test on a pas de moyen de les vérifier)
    recuperateur_donnees = gd.GestionDonnees(train_file, test_file)
    [data_train, labels_train, data_test, classes] = recuperateur_donnees.recuperer_donnees()

    # On découpe nos données pour pouvoir entrainer notre modèle
    decoupeur_donnees = dd.DecoupeDonnees(data_train, labels_train, 0.2)
    [x_train, y_train, x_test, y_test] = decoupeur_donnees.mise_en_forme_donnees()

if __name__ == "__main__":
    main()
