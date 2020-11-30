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
    # your code to run the full analysis here
    train_file = '/data/raw/train.csv'
    test_file = '/data/raw/test.csv'

    # On récupère les données d'entrainement et de test (même si test on a pas de moyen de les vérifier)
    recuperateur_donnees = gd.GestionDonnees(train_file, test_file)
    [data_train, labels_train, data_test, classes, labels_categ_train] = recuperateur_donnees.recuperer_donnees()

    # On découpe nos données pour pouvoir entrainer notre modèle
    decoupeur_donnees = dd.DecoupeDonnees(data_train, labels_train, labels_categ_train, 0.2)
    [x_train, y_train, x_test, y_test, y_categ_train] = decoupeur_donnees.mise_en_forme_donnees()

    ############################################################################################
    #Récupération des données modifié
    x_train_scaled, x_test_scaled = decoupeur_donnees.get_scaled_data(x_train, x_test)

    x_train_minmax, x_test_minmax = decoupeur_donnees.get_min_max_data(x_train, x_test)

    x_train_maxabs, x_test_maxabs = decoupeur_donnees.get_max_abs_data(x_train, x_test)

    x_train_trans, x_test_trans = decoupeur_donnees.get_quantile_data(x_train, x_test)

    x_train_gauss, x_test_gauss = decoupeur_donnees.get_gaussian_data(x_train, x_test)

    x_train_normalized, x_test_normalized = decoupeur_donnees.get_normalized_data(x_train, x_test)

    #############################################################################################
    # Entrainement de tous les modèles avec tous les types de données 
    
    print("="*30)
    print("="*30)
    print("="*30)
    print("DONNEES BRUTE : ")

    ppv = PPV.PlusProcheVoisin()
    lda = ADL.AnalyseDiscriminanteLineaire()
    rfc = RFC.RandomForest()
    svc = SVC.Noyau()
    pct = PCT.PerceptronModele()
    rn = RN.ReseauNeurone()

    """ppv.entrainement(x_train, y_train)
    lda.entrainement(x_train, y_train)
    rfc.entrainement(x_train, y_train)
    svc.entrainement(x_train, y_train)
    pct.entrainement(x_train, y_train)
    rn.entrainement(x_train, y_train)"""

    ppv.validation_croisee(x_train, y_train)
    lda.validation_croisee(x_train, y_train)
    rfc.validation_croisee(x_train, y_train)
    svc.validation_croisee(x_train, y_train)
    pct.validation_croisee(x_train, y_train)
    rn.validation_croisee(x_train, y_train)

    pred_train1 = np.array(ppv.prediction(x_train))
    pred_train2 = np.array(lda.prediction(x_train))
    pred_train3 = np.array(rfc.prediction(x_train))
    pred_train4 = np.array(svc.prediction(x_train))
    pred_train5 = np.array(pct.prediction(x_train))
    pred_train6 = np.array(rn.prediction(x_train))
    err_train1 = 0
    err_train2 = 0
    err_train3 = 0
    err_train4 = 0
    err_train5 = 0
    err_train6 = 0
    for i in range (0,len(x_train)) :
        if pred_train1[i] != y_train[i] :
            err_train1 += 1
        if pred_train2[i] != y_train[i] :
            err_train2 += 1
        if pred_train3[i] != y_train[i] :
            err_train3 += 1
        if pred_train4[i] != y_train[i] :
            err_train4 += 1
        if pred_train5[i] != y_train[i] :
            err_train5 += 1
        if pred_train6[i] != y_train[i] :
            err_train6 += 1
    err_train1 = err_train1 / len(x_train) * 100
    err_train2 = err_train2 / len(x_train) * 100
    err_train3 = err_train3 / len(x_train) * 100
    err_train4 = err_train4 / len(x_train) * 100
    err_train5 = err_train5 / len(x_train) * 100
    err_train6 = err_train6 / len(x_train) * 100

    pred_test1 = np.array(ppv.prediction(x_test))
    pred_test2 = np.array(lda.prediction(x_test))
    pred_test3 = np.array(rfc.prediction(x_test))
    pred_test4 = np.array(svc.prediction(x_test))
    pred_test5 = np.array(pct.prediction(x_test))
    pred_test6 = np.array(rn.prediction(x_test))
    err_test1 = 0
    err_test2 = 0
    err_test3 = 0
    err_test4 = 0
    err_test5 = 0
    err_test6 = 0
    for i in range (0,len(y_test)) :
        if pred_test1[i] != y_test[i] :
            err_test1 += 1
        if pred_test2[i] != y_test[i] :
            err_test2 += 1
        if pred_test3[i] != y_test[i] :
            err_test3 += 1
        if pred_test4[i] != y_test[i] :
            err_test4 += 1
        if pred_test5[i] != y_test[i] :
            err_test5 += 1
        if pred_test6[i] != y_test[i] :
            err_test6 += 1
    err_test1 = err_test1 / len(y_test) * 100
    err_test2 = err_test2 / len(y_test) * 100
    err_test3 = err_test3 / len(y_test) * 100
    err_test4 = err_test4 / len(y_test) * 100
    err_test5 = err_test5 / len(y_test) * 100
    err_test6 = err_test6 / len(y_test) * 100

    print('Plus Proches Voisin : ')
    print('Erreur train = ', err_train1, '%')
    print('Erreur test = ', err_test1, '%')
    print('algorithm : ' + ppv.algorithm + ' - weights : ' + ppv.weights + ' - n_neighbors : ' + str(ppv.n_neighbors))
    print("="*30)
    print('Analyse Discriminante Lineaire : ')
    print('Erreur train = ', err_train2, '%')
    print('Erreur test = ', err_test2, '%')
    print("solver : " + lda.solver)
    print("="*30)
    print('Random Forest Classifier: ')
    print('Erreur train = ', err_train3, '%')
    print('Erreur test = ', err_test3, '%')
    print('n_estimators : ' + str(rfc.n_estimators) + ' - max_depth : ' + str(rfc.max_depth) + ' - min_samples_leaf : ' + str(rfc.min_samples_leaf) +
            ' - max_features : ' + str(rfc.max_features) + ' - max_leaf_nodes : ' + str(rfc.max_leaf_nodes))
    print("="*30)
    print('SVC: ')
    print('Erreur train = ', err_train4, '%')
    print('Erreur test = ', err_test4, '%')
    print('kernel : ' + svc.kernel + ' - C : ' + str(svc.C) + ' - probability : ' + str(svc.probability))
    print("="*30)
    print('Perceptron: ')
    print('Erreur train = ', err_train5, '%')
    print('Erreur test = ', err_test5, '%')
    if pct.penalty != None:
        print('penalty : ' + pct.penalty + ' - alpha : ' + str(pct.alpha) + ' - max_iter : ' + str(pct.max_iter) + ' - n_iter_no_change : ' + 
                str(pct.n_iter_no_change))
    else :
        print('penalty : ' + 'None' + ' - alpha : ' + str(pct.alpha) + ' - max_iter : ' + str(pct.max_iter) + ' - n_iter_no_change : ' + 
                str(pct.n_iter_no_change))

    print("="*30)
    print('Réseau de neurone : ')
    print('Erreur train = ', err_train6, '%')
    print('Erreur test = ', err_test6, '%')
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

    ################################################################

    print("="*30)
    print("="*30)
    print("="*30)
    print("DONNEES SCALED : ")

    ppv = PPV.PlusProcheVoisin()
    lda = ADL.AnalyseDiscriminanteLineaire()
    rfc = RFC.RandomForest()
    svc = SVC.Noyau()
    pct = PCT.PerceptronModele()
    rn = RN.ReseauNeurone()

    ppv.validation_croisee(x_train_scaled, y_train)
    lda.validation_croisee(x_train_scaled, y_train)
    rfc.validation_croisee(x_train_scaled, y_train)
    svc.validation_croisee(x_train_scaled, y_train)
    pct.validation_croisee(x_train_scaled, y_train)
    rn.validation_croisee(x_train_scaled, y_train)

    pred_train1 = np.array(ppv.prediction(x_train_scaled))
    pred_train2 = np.array(lda.prediction(x_train_scaled))
    pred_train3 = np.array(rfc.prediction(x_train_scaled))
    pred_train4 = np.array(svc.prediction(x_train_scaled))
    pred_train5 = np.array(pct.prediction(x_train_scaled))
    pred_train6 = np.array(rn.prediction(x_train_scaled))
    err_train1 = 0
    err_train2 = 0
    err_train3 = 0
    err_train4 = 0
    err_train5 = 0
    err_train6 = 0
    for i in range (0,len(x_train_scaled)) :
        if pred_train1[i] != y_train[i] :
            err_train1 += 1
        if pred_train2[i] != y_train[i] :
            err_train2 += 1
        if pred_train3[i] != y_train[i] :
            err_train3 += 1
        if pred_train4[i] != y_train[i] :
            err_train4 += 1
        if pred_train5[i] != y_train[i] :
            err_train5 += 1
        if pred_train6[i] != y_train[i] :
            err_train6 += 1
    err_train1 = err_train1 / len(x_train_scaled) * 100
    err_train2 = err_train2 / len(x_train_scaled) * 100
    err_train3 = err_train3 / len(x_train_scaled) * 100
    err_train4 = err_train4 / len(x_train_scaled) * 100
    err_train5 = err_train5 / len(x_train_scaled) * 100
    err_train6 = err_train6 / len(x_train_scaled) * 100

    pred_test1 = np.array(ppv.prediction(x_test_scaled))
    pred_test2 = np.array(lda.prediction(x_test_scaled))
    pred_test3 = np.array(rfc.prediction(x_test_scaled))
    pred_test4 = np.array(svc.prediction(x_test_scaled))
    pred_test5 = np.array(pct.prediction(x_test_scaled))
    pred_test6 = np.array(rn.prediction(x_test_scaled))
    err_test1 = 0
    err_test2 = 0
    err_test3 = 0
    err_test4 = 0
    err_test5 = 0
    err_test6 = 0
    for i in range (0,len(y_test)) :
        if pred_test1[i] != y_test[i] :
            err_test1 += 1
        if pred_test2[i] != y_test[i] :
            err_test2 += 1
        if pred_test3[i] != y_test[i] :
            err_test3 += 1
        if pred_test4[i] != y_test[i] :
            err_test4 += 1
        if pred_test5[i] != y_test[i] :
            err_test5 += 1
        if pred_test6[i] != y_test[i] :
            err_test6 += 1
    err_test1 = err_test1 / len(y_test) * 100
    err_test2 = err_test2 / len(y_test) * 100
    err_test3 = err_test3 / len(y_test) * 100
    err_test4 = err_test4 / len(y_test) * 100
    err_test5 = err_test5 / len(y_test) * 100
    err_test6 = err_test6 / len(y_test) * 100

    print('Plus Proches Voisin : ')
    print('Erreur train = ', err_train1, '%')
    print('Erreur test = ', err_test1, '%')
    print('algorithm : ' + ppv.algorithm + ' - weights : ' + ppv.weights + ' - n_neighbors : ' + str(ppv.n_neighbors))
    print("="*30)
    print('Analyse Discriminante Lineaire : ')
    print('Erreur train = ', err_train2, '%')
    print('Erreur test = ', err_test2, '%')
    print("solver : " + lda.solver)
    print("="*30)
    print('Random Forest Classifier: ')
    print('Erreur train = ', err_train3, '%')
    print('Erreur test = ', err_test3, '%')
    print('n_estimators : ' + str(rfc.n_estimators) + ' - max_depth : ' + str(rfc.max_depth) + ' - min_samples_leaf : ' + str(rfc.min_samples_leaf) +
            ' - max_features : ' + str(rfc.max_features) + ' - max_leaf_nodes : ' + str(rfc.max_leaf_nodes))
    print("="*30)
    print('SVC: ')
    print('Erreur train = ', err_train4, '%')
    print('Erreur test = ', err_test4, '%')
    print('kernel : ' + svc.kernel + ' - C : ' + str(svc.C) + ' - probability : ' + str(svc.probability))
    print("="*30)
    print('Perceptron: ')
    print('Erreur train = ', err_train5, '%')
    print('Erreur test = ', err_test5, '%')
    if pct.penalty != None:
        print('penalty : ' + pct.penalty + ' - alpha : ' + str(pct.alpha) + ' - max_iter : ' + str(pct.max_iter) + ' - n_iter_no_change : ' + 
                str(pct.n_iter_no_change))
    else :
        print('penalty : ' + 'None' + ' - alpha : ' + str(pct.alpha) + ' - max_iter : ' + str(pct.max_iter) + ' - n_iter_no_change : ' + 
                str(pct.n_iter_no_change))

    print("="*30)
    print('Réseau de neurone : ')
    print('Erreur train = ', err_train6, '%')
    print('Erreur test = ', err_test6, '%')
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

    ################################################################

    print("="*30)
    print("="*30)
    print("="*30)
    print("DONNEES MIN MAX : ")

    ppv = PPV.PlusProcheVoisin()
    lda = ADL.AnalyseDiscriminanteLineaire()
    rfc = RFC.RandomForest()
    svc = SVC.Noyau()
    pct = PCT.PerceptronModele()
    rn = RN.ReseauNeurone()

    ppv.validation_croisee(x_train_minmax, y_train)
    lda.validation_croisee(x_train_minmax, y_train)
    rfc.validation_croisee(x_train_minmax, y_train)
    svc.validation_croisee(x_train_minmax, y_train)
    pct.validation_croisee(x_train_minmax, y_train)
    rn.validation_croisee(x_train_minmax, y_train)

    pred_train1 = np.array(ppv.prediction(x_train_minmax))
    pred_train2 = np.array(lda.prediction(x_train_minmax))
    pred_train3 = np.array(rfc.prediction(x_train_minmax))
    pred_train4 = np.array(svc.prediction(x_train_minmax))
    pred_train5 = np.array(pct.prediction(x_train_minmax))
    pred_train6 = np.array(rn.prediction(x_train_minmax))
    err_train1 = 0
    err_train2 = 0
    err_train3 = 0
    err_train4 = 0
    err_train5 = 0
    err_train6 = 0
    for i in range (0,len(x_train_minmax)) :
        if pred_train1[i] != y_train[i] :
            err_train1 += 1
        if pred_train2[i] != y_train[i] :
            err_train2 += 1
        if pred_train3[i] != y_train[i] :
            err_train3 += 1
        if pred_train4[i] != y_train[i] :
            err_train4 += 1
        if pred_train5[i] != y_train[i] :
            err_train5 += 1
        if pred_train6[i] != y_train[i] :
            err_train6 += 1
    err_train1 = err_train1 / len(x_train_minmax) * 100
    err_train2 = err_train2 / len(x_train_minmax) * 100
    err_train3 = err_train3 / len(x_train_minmax) * 100
    err_train4 = err_train4 / len(x_train_minmax) * 100
    err_train5 = err_train5 / len(x_train_minmax) * 100
    err_train6 = err_train6 / len(x_train_minmax) * 100

    pred_test1 = np.array(ppv.prediction(x_test_minmax))
    pred_test2 = np.array(lda.prediction(x_test_minmax))
    pred_test3 = np.array(rfc.prediction(x_test_minmax))
    pred_test4 = np.array(svc.prediction(x_test_minmax))
    pred_test5 = np.array(pct.prediction(x_test_minmax))
    pred_test6 = np.array(rn.prediction(x_test_minmax))
    err_test1 = 0
    err_test2 = 0
    err_test3 = 0
    err_test4 = 0
    err_test5 = 0
    err_test6 = 0
    for i in range (0,len(y_test)) :
        if pred_test1[i] != y_test[i] :
            err_test1 += 1
        if pred_test2[i] != y_test[i] :
            err_test2 += 1
        if pred_test3[i] != y_test[i] :
            err_test3 += 1
        if pred_test4[i] != y_test[i] :
            err_test4 += 1
        if pred_test5[i] != y_test[i] :
            err_test5 += 1
        if pred_test6[i] != y_test[i] :
            err_test6 += 1
    err_test1 = err_test1 / len(y_test) * 100
    err_test2 = err_test2 / len(y_test) * 100
    err_test3 = err_test3 / len(y_test) * 100
    err_test4 = err_test4 / len(y_test) * 100
    err_test5 = err_test5 / len(y_test) * 100
    err_test6 = err_test6 / len(y_test) * 100

    print('Plus Proches Voisin : ')
    print('Erreur train = ', err_train1, '%')
    print('Erreur test = ', err_test1, '%')
    print('algorithm : ' + ppv.algorithm + ' - weights : ' + ppv.weights + ' - n_neighbors : ' + str(ppv.n_neighbors))
    print("="*30)
    print('Analyse Discriminante Lineaire : ')
    print('Erreur train = ', err_train2, '%')
    print('Erreur test = ', err_test2, '%')
    print("solver : " + lda.solver)
    print("="*30)
    print('Random Forest Classifier: ')
    print('Erreur train = ', err_train3, '%')
    print('Erreur test = ', err_test3, '%')
    print('n_estimators : ' + str(rfc.n_estimators) + ' - max_depth : ' + str(rfc.max_depth) + ' - min_samples_leaf : ' + str(rfc.min_samples_leaf) +
            ' - max_features : ' + str(rfc.max_features) + ' - max_leaf_nodes : ' + str(rfc.max_leaf_nodes))
    print("="*30)
    print('SVC: ')
    print('Erreur train = ', err_train4, '%')
    print('Erreur test = ', err_test4, '%')
    print('kernel : ' + svc.kernel + ' - C : ' + str(svc.C) + ' - probability : ' + str(svc.probability))
    print("="*30)
    print('Perceptron: ')
    print('Erreur train = ', err_train5, '%')
    print('Erreur test = ', err_test5, '%')
    if pct.penalty != None:
        print('penalty : ' + pct.penalty + ' - alpha : ' + str(pct.alpha) + ' - max_iter : ' + str(pct.max_iter) + ' - n_iter_no_change : ' + 
                str(pct.n_iter_no_change))
    else :
        print('penalty : ' + 'None' + ' - alpha : ' + str(pct.alpha) + ' - max_iter : ' + str(pct.max_iter) + ' - n_iter_no_change : ' + 
                str(pct.n_iter_no_change))

    print("="*30)
    print('Réseau de neurone : ')
    print('Erreur train = ', err_train6, '%')
    print('Erreur test = ', err_test6, '%')
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

    ################################################################

    print("="*30)
    print("="*30)
    print("="*30)
    print("DONNEES MAX ABS : ")

    ppv = PPV.PlusProcheVoisin()
    lda = ADL.AnalyseDiscriminanteLineaire()
    rfc = RFC.RandomForest()
    svc = SVC.Noyau()
    pct = PCT.PerceptronModele()
    rn = RN.ReseauNeurone()

    ppv.validation_croisee(x_train_maxabs, y_train)
    lda.validation_croisee(x_train_maxabs, y_train)
    rfc.validation_croisee(x_train_maxabs, y_train)
    svc.validation_croisee(x_train_maxabs, y_train)
    pct.validation_croisee(x_train_maxabs, y_train)
    rn.validation_croisee(x_train_maxabs, y_train)

    pred_train1 = np.array(ppv.prediction(x_train_maxabs))
    pred_train2 = np.array(lda.prediction(x_train_maxabs))
    pred_train3 = np.array(rfc.prediction(x_train_maxabs))
    pred_train4 = np.array(svc.prediction(x_train_maxabs))
    pred_train5 = np.array(pct.prediction(x_train_maxabs))
    pred_train6 = np.array(rn.prediction(x_train_maxabs))
    err_train1 = 0
    err_train2 = 0
    err_train3 = 0
    err_train4 = 0
    err_train5 = 0
    err_train6 = 0
    for i in range (0,len(x_train_maxabs)) :
        if pred_train1[i] != y_train[i] :
            err_train1 += 1
        if pred_train2[i] != y_train[i] :
            err_train2 += 1
        if pred_train3[i] != y_train[i] :
            err_train3 += 1
        if pred_train4[i] != y_train[i] :
            err_train4 += 1
        if pred_train5[i] != y_train[i] :
            err_train5 += 1
        if pred_train6[i] != y_train[i] :
            err_train6 += 1
    err_train1 = err_train1 / len(x_train_maxabs) * 100
    err_train2 = err_train2 / len(x_train_maxabs) * 100
    err_train3 = err_train3 / len(x_train_maxabs) * 100
    err_train4 = err_train4 / len(x_train_maxabs) * 100
    err_train5 = err_train5 / len(x_train_maxabs) * 100
    err_train6 = err_train6 / len(x_train_maxabs) * 100

    pred_test1 = np.array(ppv.prediction(x_test_maxabs))
    pred_test2 = np.array(lda.prediction(x_test_maxabs))
    pred_test3 = np.array(rfc.prediction(x_test_maxabs))
    pred_test4 = np.array(svc.prediction(x_test_maxabs))
    pred_test5 = np.array(pct.prediction(x_test_maxabs))
    pred_test6 = np.array(rn.prediction(x_test_maxabs))
    err_test1 = 0
    err_test2 = 0
    err_test3 = 0
    err_test4 = 0
    err_test5 = 0
    err_test6 = 0
    for i in range (0,len(y_test)) :
        if pred_test1[i] != y_test[i] :
            err_test1 += 1
        if pred_test2[i] != y_test[i] :
            err_test2 += 1
        if pred_test3[i] != y_test[i] :
            err_test3 += 1
        if pred_test4[i] != y_test[i] :
            err_test4 += 1
        if pred_test5[i] != y_test[i] :
            err_test5 += 1
        if pred_test6[i] != y_test[i] :
            err_test6 += 1
    err_test1 = err_test1 / len(y_test) * 100
    err_test2 = err_test2 / len(y_test) * 100
    err_test3 = err_test3 / len(y_test) * 100
    err_test4 = err_test4 / len(y_test) * 100
    err_test5 = err_test5 / len(y_test) * 100
    err_test6 = err_test6 / len(y_test) * 100

    print('Plus Proches Voisin : ')
    print('Erreur train = ', err_train1, '%')
    print('Erreur test = ', err_test1, '%')
    print('algorithm : ' + ppv.algorithm + ' - weights : ' + ppv.weights + ' - n_neighbors : ' + str(ppv.n_neighbors))
    print("="*30)
    print('Analyse Discriminante Lineaire : ')
    print('Erreur train = ', err_train2, '%')
    print('Erreur test = ', err_test2, '%')
    print("solver : " + lda.solver)
    print("="*30)
    print('Random Forest Classifier: ')
    print('Erreur train = ', err_train3, '%')
    print('Erreur test = ', err_test3, '%')
    print('n_estimators : ' + str(rfc.n_estimators) + ' - max_depth : ' + str(rfc.max_depth) + ' - min_samples_leaf : ' + str(rfc.min_samples_leaf) +
            ' - max_features : ' + str(rfc.max_features) + ' - max_leaf_nodes : ' + str(rfc.max_leaf_nodes))
    print("="*30)
    print('SVC: ')
    print('Erreur train = ', err_train4, '%')
    print('Erreur test = ', err_test4, '%')
    print('kernel : ' + svc.kernel + ' - C : ' + str(svc.C) + ' - probability : ' + str(svc.probability))
    print("="*30)
    print('Perceptron: ')
    print('Erreur train = ', err_train5, '%')
    print('Erreur test = ', err_test5, '%')
    if pct.penalty != None:
        print('penalty : ' + pct.penalty + ' - alpha : ' + str(pct.alpha) + ' - max_iter : ' + str(pct.max_iter) + ' - n_iter_no_change : ' + 
                str(pct.n_iter_no_change))
    else :
        print('penalty : ' + 'None' + ' - alpha : ' + str(pct.alpha) + ' - max_iter : ' + str(pct.max_iter) + ' - n_iter_no_change : ' + 
                str(pct.n_iter_no_change))

    print("="*30)
    print('Réseau de neurone : ')
    print('Erreur train = ', err_train6, '%')
    print('Erreur test = ', err_test6, '%')
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

    ################################################################

    print("="*30)
    print("="*30)
    print("="*30)
    print("DONNEES QUANTILE : ")

    ppv = PPV.PlusProcheVoisin()
    lda = ADL.AnalyseDiscriminanteLineaire()
    rfc = RFC.RandomForest()
    svc = SVC.Noyau()
    pct = PCT.PerceptronModele()
    rn = RN.ReseauNeurone()

    ppv.validation_croisee(x_train_trans, y_train)
    lda.validation_croisee(x_train_trans, y_train)
    rfc.validation_croisee(x_train_trans, y_train)
    svc.validation_croisee(x_train_trans, y_train)
    pct.validation_croisee(x_train_trans, y_train)
    rn.validation_croisee(x_train_trans, y_train)

    pred_train1 = np.array(ppv.prediction(x_train_trans))
    pred_train2 = np.array(lda.prediction(x_train_trans))
    pred_train3 = np.array(rfc.prediction(x_train_trans))
    pred_train4 = np.array(svc.prediction(x_train_trans))
    pred_train5 = np.array(pct.prediction(x_train_trans))
    pred_train6 = np.array(rn.prediction(x_train_trans))
    err_train1 = 0
    err_train2 = 0
    err_train3 = 0
    err_train4 = 0
    err_train5 = 0
    err_train6 = 0
    for i in range (0,len(x_train_trans)) :
        if pred_train1[i] != y_train[i] :
            err_train1 += 1
        if pred_train2[i] != y_train[i] :
            err_train2 += 1
        if pred_train3[i] != y_train[i] :
            err_train3 += 1
        if pred_train4[i] != y_train[i] :
            err_train4 += 1
        if pred_train5[i] != y_train[i] :
            err_train5 += 1
        if pred_train6[i] != y_train[i] :
            err_train6 += 1
    err_train1 = err_train1 / len(x_train_trans) * 100
    err_train2 = err_train2 / len(x_train_trans) * 100
    err_train3 = err_train3 / len(x_train_trans) * 100
    err_train4 = err_train4 / len(x_train_trans) * 100
    err_train5 = err_train5 / len(x_train_trans) * 100
    err_train6 = err_train6 / len(x_train_trans) * 100

    pred_test1 = np.array(ppv.prediction(x_test_trans))
    pred_test2 = np.array(lda.prediction(x_test_trans))
    pred_test3 = np.array(rfc.prediction(x_test_trans))
    pred_test4 = np.array(svc.prediction(x_test_trans))
    pred_test5 = np.array(pct.prediction(x_test_trans))
    pred_test6 = np.array(rn.prediction(x_test_trans))
    err_test1 = 0
    err_test2 = 0
    err_test3 = 0
    err_test4 = 0
    err_test5 = 0
    err_test6 = 0
    for i in range (0,len(y_test)) :
        if pred_test1[i] != y_test[i] :
            err_test1 += 1
        if pred_test2[i] != y_test[i] :
            err_test2 += 1
        if pred_test3[i] != y_test[i] :
            err_test3 += 1
        if pred_test4[i] != y_test[i] :
            err_test4 += 1
        if pred_test5[i] != y_test[i] :
            err_test5 += 1
        if pred_test6[i] != y_test[i] :
            err_test6 += 1
    err_test1 = err_test1 / len(y_test) * 100
    err_test2 = err_test2 / len(y_test) * 100
    err_test3 = err_test3 / len(y_test) * 100
    err_test4 = err_test4 / len(y_test) * 100
    err_test5 = err_test5 / len(y_test) * 100
    err_test6 = err_test6 / len(y_test) * 100

    print('Plus Proches Voisin : ')
    print('Erreur train = ', err_train1, '%')
    print('Erreur test = ', err_test1, '%')
    print('algorithm : ' + ppv.algorithm + ' - weights : ' + ppv.weights + ' - n_neighbors : ' + str(ppv.n_neighbors))
    print("="*30)
    print('Analyse Discriminante Lineaire : ')
    print('Erreur train = ', err_train2, '%')
    print('Erreur test = ', err_test2, '%')
    print("solver : " + lda.solver)
    print("="*30)
    print('Random Forest Classifier: ')
    print('Erreur train = ', err_train3, '%')
    print('Erreur test = ', err_test3, '%')
    print('n_estimators : ' + str(rfc.n_estimators) + ' - max_depth : ' + str(rfc.max_depth) + ' - min_samples_leaf : ' + str(rfc.min_samples_leaf) +
            ' - max_features : ' + str(rfc.max_features) + ' - max_leaf_nodes : ' + str(rfc.max_leaf_nodes))
    print("="*30)
    print('SVC: ')
    print('Erreur train = ', err_train4, '%')
    print('Erreur test = ', err_test4, '%')
    print('kernel : ' + svc.kernel + ' - C : ' + str(svc.C) + ' - probability : ' + str(svc.probability))
    print("="*30)
    print('Perceptron: ')
    print('Erreur train = ', err_train5, '%')
    print('Erreur test = ', err_test5, '%')
    if pct.penalty != None:
        print('penalty : ' + pct.penalty + ' - alpha : ' + str(pct.alpha) + ' - max_iter : ' + str(pct.max_iter) + ' - n_iter_no_change : ' + 
                str(pct.n_iter_no_change))
    else :
        print('penalty : ' + 'None' + ' - alpha : ' + str(pct.alpha) + ' - max_iter : ' + str(pct.max_iter) + ' - n_iter_no_change : ' + 
                str(pct.n_iter_no_change))

    print("="*30)
    print('Réseau de neurone : ')
    print('Erreur train = ', err_train6, '%')
    print('Erreur test = ', err_test6, '%')
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

    ################################################################

    print("="*30)
    print("="*30)
    print("="*30)
    print("DONNEES GAUSSIENNE : ")

    ppv = PPV.PlusProcheVoisin()
    lda = ADL.AnalyseDiscriminanteLineaire()
    rfc = RFC.RandomForest()
    svc = SVC.Noyau()
    pct = PCT.PerceptronModele()
    rn = RN.ReseauNeurone()

    ppv.validation_croisee(x_train_gauss, y_train)
    lda.validation_croisee(x_train_gauss, y_train)
    rfc.validation_croisee(x_train_gauss, y_train)
    svc.validation_croisee(x_train_gauss, y_train)
    pct.validation_croisee(x_train_gauss, y_train)
    rn.validation_croisee(x_train_gauss, y_train)

    pred_train1 = np.array(ppv.prediction(x_train_gauss))
    pred_train2 = np.array(lda.prediction(x_train_gauss))
    pred_train3 = np.array(rfc.prediction(x_train_gauss))
    pred_train4 = np.array(svc.prediction(x_train_gauss))
    pred_train5 = np.array(pct.prediction(x_train_gauss))
    pred_train6 = np.array(rn.prediction(x_train_gauss))
    err_train1 = 0
    err_train2 = 0
    err_train3 = 0
    err_train4 = 0
    err_train5 = 0
    err_train6 = 0
    for i in range (0,len(x_train_gauss)) :
        if pred_train1[i] != y_train[i] :
            err_train1 += 1
        if pred_train2[i] != y_train[i] :
            err_train2 += 1
        if pred_train3[i] != y_train[i] :
            err_train3 += 1
        if pred_train4[i] != y_train[i] :
            err_train4 += 1
        if pred_train5[i] != y_train[i] :
            err_train5 += 1
        if pred_train6[i] != y_train[i] :
            err_train6 += 1
    err_train1 = err_train1 / len(x_train_gauss) * 100
    err_train2 = err_train2 / len(x_train_gauss) * 100
    err_train3 = err_train3 / len(x_train_gauss) * 100
    err_train4 = err_train4 / len(x_train_gauss) * 100
    err_train5 = err_train5 / len(x_train_gauss) * 100
    err_train6 = err_train6 / len(x_train_gauss) * 100

    pred_test1 = np.array(ppv.prediction(x_test_gauss))
    pred_test2 = np.array(lda.prediction(x_test_gauss))
    pred_test3 = np.array(rfc.prediction(x_test_gauss))
    pred_test4 = np.array(svc.prediction(x_test_gauss))
    pred_test5 = np.array(pct.prediction(x_test_gauss))
    pred_test6 = np.array(rn.prediction(x_test_gauss))
    err_test1 = 0
    err_test2 = 0
    err_test3 = 0
    err_test4 = 0
    err_test5 = 0
    err_test6 = 0
    for i in range (0,len(y_test)) :
        if pred_test1[i] != y_test[i] :
            err_test1 += 1
        if pred_test2[i] != y_test[i] :
            err_test2 += 1
        if pred_test3[i] != y_test[i] :
            err_test3 += 1
        if pred_test4[i] != y_test[i] :
            err_test4 += 1
        if pred_test5[i] != y_test[i] :
            err_test5 += 1
        if pred_test6[i] != y_test[i] :
            err_test6 += 1
    err_test1 = err_test1 / len(y_test) * 100
    err_test2 = err_test2 / len(y_test) * 100
    err_test3 = err_test3 / len(y_test) * 100
    err_test4 = err_test4 / len(y_test) * 100
    err_test5 = err_test5 / len(y_test) * 100
    err_test6 = err_test6 / len(y_test) * 100

    print('Plus Proches Voisin : ')
    print('Erreur train = ', err_train1, '%')
    print('Erreur test = ', err_test1, '%')
    print('algorithm : ' + ppv.algorithm + ' - weights : ' + ppv.weights + ' - n_neighbors : ' + str(ppv.n_neighbors))
    print("="*30)
    print('Analyse Discriminante Lineaire : ')
    print('Erreur train = ', err_train2, '%')
    print('Erreur test = ', err_test2, '%')
    print("solver : " + lda.solver)
    print("="*30)
    print('Random Forest Classifier: ')
    print('Erreur train = ', err_train3, '%')
    print('Erreur test = ', err_test3, '%')
    print('n_estimators : ' + str(rfc.n_estimators) + ' - max_depth : ' + str(rfc.max_depth) + ' - min_samples_leaf : ' + str(rfc.min_samples_leaf) +
            ' - max_features : ' + str(rfc.max_features) + ' - max_leaf_nodes : ' + str(rfc.max_leaf_nodes))
    print("="*30)
    print('SVC: ')
    print('Erreur train = ', err_train4, '%')
    print('Erreur test = ', err_test4, '%')
    print('kernel : ' + svc.kernel + ' - C : ' + str(svc.C) + ' - probability : ' + str(svc.probability))
    print("="*30)
    print('Perceptron: ')
    print('Erreur train = ', err_train5, '%')
    print('Erreur test = ', err_test5, '%')
    if pct.penalty != None:
        print('penalty : ' + pct.penalty + ' - alpha : ' + str(pct.alpha) + ' - max_iter : ' + str(pct.max_iter) + ' - n_iter_no_change : ' + 
                str(pct.n_iter_no_change))
    else :
        print('penalty : ' + 'None' + ' - alpha : ' + str(pct.alpha) + ' - max_iter : ' + str(pct.max_iter) + ' - n_iter_no_change : ' + 
                str(pct.n_iter_no_change))

    print("="*30)
    print('Réseau de neurone : ')
    print('Erreur train = ', err_train6, '%')
    print('Erreur test = ', err_test6, '%')
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

    ################################################################

    print("="*30)
    print("="*30)
    print("="*30)
    print("DONNEES NORMALIZE : ")

    ppv = PPV.PlusProcheVoisin()
    lda = ADL.AnalyseDiscriminanteLineaire()
    rfc = RFC.RandomForest()
    svc = SVC.Noyau()
    pct = PCT.PerceptronModele()
    rn = RN.ReseauNeurone()

    ppv.validation_croisee(x_train_normalized, y_train)
    lda.validation_croisee(x_train_normalized, y_train)
    rfc.validation_croisee(x_train_normalized, y_train)
    svc.validation_croisee(x_train_normalized, y_train)
    pct.validation_croisee(x_train_normalized, y_train)
    rn.validation_croisee(x_train_normalized, y_train)

    pred_train1 = np.array(ppv.prediction(x_train_normalized))
    pred_train2 = np.array(lda.prediction(x_train_normalized))
    pred_train3 = np.array(rfc.prediction(x_train_normalized))
    pred_train4 = np.array(svc.prediction(x_train_normalized))
    pred_train5 = np.array(pct.prediction(x_train_normalized))
    pred_train6 = np.array(rn.prediction(x_train_normalized))
    err_train1 = 0
    err_train2 = 0
    err_train3 = 0
    err_train4 = 0
    err_train5 = 0
    err_train6 = 0
    for i in range (0,len(x_train_normalized)) :
        if pred_train1[i] != y_train[i] :
            err_train1 += 1
        if pred_train2[i] != y_train[i] :
            err_train2 += 1
        if pred_train3[i] != y_train[i] :
            err_train3 += 1
        if pred_train4[i] != y_train[i] :
            err_train4 += 1
        if pred_train5[i] != y_train[i] :
            err_train5 += 1
        if pred_train6[i] != y_train[i] :
            err_train6 += 1
    err_train1 = err_train1 / len(x_train_normalized) * 100
    err_train2 = err_train2 / len(x_train_normalized) * 100
    err_train3 = err_train3 / len(x_train_normalized) * 100
    err_train4 = err_train4 / len(x_train_normalized) * 100
    err_train5 = err_train5 / len(x_train_normalized) * 100
    err_train6 = err_train6 / len(x_train_normalized) * 100

    pred_test1 = np.array(ppv.prediction(x_test_normalized))
    pred_test2 = np.array(lda.prediction(x_test_normalized))
    pred_test3 = np.array(rfc.prediction(x_test_normalized))
    pred_test4 = np.array(svc.prediction(x_test_normalized))
    pred_test5 = np.array(pct.prediction(x_test_normalized))
    pred_test6 = np.array(rn.prediction(x_test_normalized))
    err_test1 = 0
    err_test2 = 0
    err_test3 = 0
    err_test4 = 0
    err_test5 = 0
    err_test6 = 0
    for i in range (0,len(y_test)) :
        if pred_test1[i] != y_test[i] :
            err_test1 += 1
        if pred_test2[i] != y_test[i] :
            err_test2 += 1
        if pred_test3[i] != y_test[i] :
            err_test3 += 1
        if pred_test4[i] != y_test[i] :
            err_test4 += 1
        if pred_test5[i] != y_test[i] :
            err_test5 += 1
        if pred_test6[i] != y_test[i] :
            err_test6 += 1
    err_test1 = err_test1 / len(y_test) * 100
    err_test2 = err_test2 / len(y_test) * 100
    err_test3 = err_test3 / len(y_test) * 100
    err_test4 = err_test4 / len(y_test) * 100
    err_test5 = err_test5 / len(y_test) * 100
    err_test6 = err_test6 / len(y_test) * 100

    print('Plus Proches Voisin : ')
    print('Erreur train = ', err_train1, '%')
    print('Erreur test = ', err_test1, '%')
    print('algorithm : ' + ppv.algorithm + ' - weights : ' + ppv.weights + ' - n_neighbors : ' + str(ppv.n_neighbors))
    print("="*30)
    print('Analyse Discriminante Lineaire : ')
    print('Erreur train = ', err_train2, '%')
    print('Erreur test = ', err_test2, '%')
    print("solver : " + lda.solver)
    print("="*30)
    print('Random Forest Classifier: ')
    print('Erreur train = ', err_train3, '%')
    print('Erreur test = ', err_test3, '%')
    print('n_estimators : ' + str(rfc.n_estimators) + ' - max_depth : ' + str(rfc.max_depth) + ' - min_samples_leaf : ' + str(rfc.min_samples_leaf) +
            ' - max_features : ' + str(rfc.max_features) + ' - max_leaf_nodes : ' + str(rfc.max_leaf_nodes))
    print("="*30)
    print('SVC: ')
    print('Erreur train = ', err_train4, '%')
    print('Erreur test = ', err_test4, '%')
    print('kernel : ' + svc.kernel + ' - C : ' + str(svc.C) + ' - probability : ' + str(svc.probability))
    print("="*30)
    print('Perceptron: ')
    print('Erreur train = ', err_train5, '%')
    print('Erreur test = ', err_test5, '%')
    if pct.penalty != None:
        print('penalty : ' + pct.penalty + ' - alpha : ' + str(pct.alpha) + ' - max_iter : ' + str(pct.max_iter) + ' - n_iter_no_change : ' + 
                str(pct.n_iter_no_change))
    else :
        print('penalty : ' + 'None' + ' - alpha : ' + str(pct.alpha) + ' - max_iter : ' + str(pct.max_iter) + ' - n_iter_no_change : ' + 
                str(pct.n_iter_no_change))

    print("="*30)
    print('Réseau de neurone : ')
    print('Erreur train = ', err_train6, '%')
    print('Erreur test = ', err_test6, '%')
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

    ############################################################################################
    # Utilisation de la méthode en deux étapes pour faire la prédiction, avec des données prétraité 
    # avec la méthode du Min_Max ici 

    print("="*30)
    print("="*30)
    print("="*30)
    print("DONNEES Min Max : ")

    2_step_pred = TSP.BossFinal()
    2_step_pred.entrainementModele(x_train_minmax, y_train, y_categ_train)

    pred_train = 2_step_pred.predict(x_train_minmax)

    err_train = []
    for i in range (0,6):
        tmp = []
        for j in range(0,6):
            err = 0 
            for k in range (0,len(x_train_minmax)) :
                if pred_train[i][j][k] != y_train[k] :
                    err += 1
            err = err / len(x_train_minmax) * 100
            tmp.append(err)
        err_train.append(tmp)

    pred_test = 2_step_pred.predict(x_test_minmax)

    err_test = []
    for i in range (0,6):
        tmp = []
        for j in range(0,6):
            err = 0 
            for k in range (0,len(x_test_minmax)) :
                if pred_test[i][j][k] != y_test[k] :
                    err += 1
            err = err / len(x_test_minmax) * 100
            tmp.append(err)
        err_test.append(tmp)

    print("="*30)
    print("="*30)
    for i in range (0,6):
        for j in range (0,6):
            print("combinaison : " + str(i) + " - " + str(j))
            print('Erreur train = ', err_train[i][j], '%')
            print('Erreur test = ', err_test[i][j], '%')
            print("="*30)

if __name__ == "__main__":
    build_project()