# -*- coding: utf-8 -*-
"""
Execution dans un terminal

Exemple:
    python3 run.py PPV None 0
"""

#####
# Félix Gaucher (gauf2611)
# Adrien Verdier (vera2704)
###

import numpy as np
import sys
import src.data.gestion_donnees as gd
import src.features.decoupage_donnees as dd
import src.models.Plus_Proche_Voisin as PPV
import src.models.Analyse_discriminante_lineaire as ADL
import src.models.random_forest as RFC
import src.models.noyau as SVC
import src.models.perceptron as PCT
import src.models.reseau_neurone as RN
import src.two_step_prediction as TSP

def build_project():

    if len(sys.argv) < 4:
        usage = "\n Usage: python3 run.py modele preprocessing grid_search\
        \n\n\t modele: PPV, ADL, RFC, SVC, PCT, RN, DoubleSearch\
        \n\t preprocessing: None, Scaled, MinMax, MaxAbs, Quantile, Gaussian, Normalize\
        \n\t grid_search: 0: pas de grid search, 1: grid search\n"
        print(usage)
        return

    modele = sys.argv[1]
    preprocessing = sys.argv[2]
    gs = bool(int(sys.argv[3]))

    # Nos deux fichiers de données
    train_file = '/data/raw/train.csv'
    test_file = '/data/raw/test.csv'

    # On récupère les données d'entrainement et de test 
    recuperateur_donnees = gd.GestionDonnees(train_file, test_file)
    [data_train, labels_train, data_test, classes, labels_categ_train] = recuperateur_donnees.recuperer_donnees()

    # On découpe nos données pour pouvoir entrainer notre modèle
    decoupeur_donnees = dd.DecoupeDonnees(data_train, labels_train, labels_categ_train, 0.2)
    [x_train, y_train, x_test, y_test, y_categ_train] = decoupeur_donnees.mise_en_forme_donnees()

    # Preprocessing sur nos données (aucun si None)
    if preprocessing == "Scaled" :
        x_train, x_test = decoupeur_donnees.get_scaled_data(x_train, x_test)

    elif preprocessing == "MinMax" :
        x_train, x_test = decoupeur_donnees.get_min_max_data(x_train, x_test)

    elif preprocessing == "MaxAbs" :
        x_train, x_test = decoupeur_donnees.get_max_abs_data(x_train, x_test)

    elif preprocessing == "Quantile" :
        x_train, x_test = decoupeur_donnees.get_quantile_data(x_train, x_test)

    elif preprocessing == "Gaussian" :
        x_train, x_test = decoupeur_donnees.get_gaussian_data(x_train, x_test)

    elif preprocessing == "Normalize" :
        x_train, x_test = decoupeur_donnees.get_normalized_data(x_train, x_test)

    # Entrainement du modèle choisi par l'utilisateur
    if modele == "PPV" :
        ppv = PPV.PlusProcheVoisin()

        if gs is True:
            ppv.validation_croisee(x_train, y_train)
        else :
            ppv.entrainement(x_train, y_train)

        pred_train = np.array(ppv.prediction(x_train))
        err_train = 0
        for i in range (0,len(x_train)) :
            if pred_train[i] != y_train[i] :
                err_train += 1
        err_train = err_train / len(x_train) * 100

        pred_test = np.array(ppv.prediction(x_test))
        err_test = 0
        for i in range (0,len(y_test)) :
            if pred_test[i] != y_test[i] :
                err_test += 1
        err_test = err_test / len(y_test) * 100

        print('Plus Proches Voisin : ')
        print('Erreur train = ', err_train, '%')
        print('Erreur test = ', err_test, '%')
        print('algorithm : ' + ppv.algorithm + ' - weights : ' + ppv.weights + ' - n_neighbors : ' + str(ppv.n_neighbors))

    elif modele == "ADL" :
        lda = ADL.AnalyseDiscriminanteLineaire()

        if gs is True:
            lda.validation_croisee(x_train, y_train)
        else :
            lda.entrainement(x_train, y_train)

        pred_train = np.array(lda.prediction(x_train))
        err_train = 0
        for i in range (0,len(x_train)) :
            if pred_train[i] != y_train[i] :
                err_train += 1
        err_train = err_train / len(x_train) * 100

        pred_test = np.array(lda.prediction(x_test))
        err_test = 0
        for i in range (0,len(y_test)) :
            if pred_test[i] != y_test[i] :
                err_test += 1
        err_test = err_test / len(y_test) * 100

        print('Analyse Discriminante Lineaire : ')
        print('Erreur train = ', err_train, '%')
        print('Erreur test = ', err_test, '%')
        print("solver : " + lda.solver)

    elif modele == "RFC" :
        rfc = RFC.RandomForest()

        if gs is True:
            rfc.validation_croisee(x_train, y_train)
        else :
            rfc.entrainement(x_train, y_train)

        pred_train = np.array(rfc.prediction(x_train))
        err_train = 0
        for i in range (0,len(x_train)) :
            if pred_train[i] != y_train[i] :
                err_train += 1
        err_train = err_train / len(x_train) * 100

        pred_test = np.array(rfc.prediction(x_test))
        err_test = 0
        for i in range (0,len(y_test)) :
            if pred_test[i] != y_test[i] :
                err_test += 1
        err_test = err_test / len(y_test) * 100

        print('Random Forest Classifier: ')
        print('Erreur train = ', err_train, '%')
        print('Erreur test = ', err_test, '%')
        print('n_estimators : ' + str(rfc.n_estimators) + ' - max_depth : ' + str(rfc.max_depth) + ' - min_samples_leaf : ' + str(rfc.min_samples_leaf) +
                ' - max_features : ' + str(rfc.max_features) + ' - max_leaf_nodes : ' + str(rfc.max_leaf_nodes))

    elif modele == "SVC" :
        svc = SVC.Noyau()

        if gs is True:
            svc.validation_croisee(x_train, y_train)
        else :
            svc.entrainement(x_train, y_train)

        pred_train = np.array(svc.prediction(x_train))
        err_train = 0
        for i in range (0,len(x_train)) :
            if pred_train[i] != y_train[i] :
                err_train += 1
        err_train = err_train / len(x_train) * 100

        pred_test = np.array(svc.prediction(x_test))
        err_test = 0
        for i in range (0,len(y_test)) :
            if pred_test[i] != y_test[i] :
                err_test += 1
        err_test = err_test / len(y_test) * 100

        print('SVC: ')
        print('Erreur train = ', err_train, '%')
        print('Erreur test = ', err_test, '%')
        print('kernel : ' + svc.kernel + ' - C : ' + str(svc.C) + ' - probability : ' + str(svc.probability))

    elif modele == "PCT" :
        pct = PCT.PerceptronModele()

        if gs is True:
            pct.validation_croisee(x_train, y_train)
        else :
            pct.entrainement(x_train, y_train)

        pred_train = np.array(pct.prediction(x_train))
        err_train = 0
        for i in range (0,len(x_train)) :
            if pred_train[i] != y_train[i] :
                err_train += 1
        err_train = err_train / len(x_train) * 100

        pred_test = np.array(pct.prediction(x_test))
        err_test = 0
        for i in range (0,len(y_test)) :
            if pred_test[i] != y_test[i] :
                err_test += 1
        err_test = err_test / len(y_test) * 100

        print('Perceptron: ')
        print('Erreur train = ', err_train, '%')
        print('Erreur test = ', err_test, '%')
        if pct.penalty != None:
            print('penalty : ' + pct.penalty + ' - alpha : ' + str(pct.alpha) + ' - max_iter : ' + str(pct.max_iter) + ' - n_iter_no_change : ' + 
                    str(pct.n_iter_no_change))
        else :
            print('penalty : ' + 'None' + ' - alpha : ' + str(pct.alpha) + ' - max_iter : ' + str(pct.max_iter) + ' - n_iter_no_change : ' + 
                    str(pct.n_iter_no_change))

    elif modele == "RN" :
        rn = RN.ReseauNeurone()

        if gs is True:
            rn.validation_croisee(x_train, y_train)
        else :
            rn.entrainement(x_train, y_train)

        pred_train = np.array(rn.prediction(x_train))
        err_train = 0
        for i in range (0,len(x_train)) :
            if pred_train[i] != y_train[i] :
                err_train += 1
        err_train = err_train / len(x_train) * 100

        pred_test = np.array(rn.prediction(x_test))
        err_test = 0
        for i in range (0,len(y_test)) :
            if pred_test[i] != y_test[i] :
                err_test += 1
        err_test = err_test / len(y_test) * 100

        print('Réseau de neurone : ')
        print('Erreur train = ', err_train, '%')
        print('Erreur test = ', err_test, '%')
        print('hidden_layer_sizes : ' + str(rn.hidden_layer_sizes) + 
                ' - activation : ' + rn.activation + 
                ' - solver : ' + rn.solver + 
                ' - alpha : ' + str(rn.alpha) + 
                ' - learning_rate : ' + rn.learning_rate + 
                ' - learning_rate_init : ' + str(rn.learning_rate_init) + 
                ' - power_t : ' + str(rn.power_t) + 
                ' - max_iter : ' + str(rn.max_iter) + 
                ' - momentum : ' + str(rn.momentum) + 
                ' - beta_1 : ' + str(rn.beta_1) + 
                ' - beta_2 : ' + str(rn.beta_2) + 
                ' - n_iter_no_change : ' + str(rn.n_iter_no_change) + 
                ' - max_fun : ' + str(rn.max_fun))

    elif modele == "DoubleSearch" : 
        2_step_pred = TSP.BossFinal()
        2_step_pred.entrainementModele(x_train, y_train, y_categ_train)

        pred_train = 2_step_pred.predict(x_train)
        err_train = []
        for i in range (0,6):
            tmp = []
            for j in range(0,6):
                err = 0 
                for k in range (0,len(x_train)) :
                    if pred_train[i][j][k] != y_train[k] :
                        err += 1
                err = err / len(x_train) * 100
                tmp.append(err)
            err_train.append(tmp)

        pred_test = 2_step_pred.predict(x_test)
        err_test = []
        for i in range (0,6):
            tmp = []
            for j in range(0,6):
                err = 0 
                for k in range (0,len(x_test)) :
                    if pred_test[i][j][k] != y_test[k] :
                        err += 1
                err = err / len(x_test) * 100
                tmp.append(err)
            err_test.append(tmp)

        for i in range (0,6):
            for j in range (0,6):
                print("combinaison : " + str(i) + " - " + str(j))
                print('Erreur train = ', err_train[i][j], '%')
                print('Erreur test = ', err_test[i][j], '%')
                print("="*30)


if __name__ == "__main__":
    build_project()