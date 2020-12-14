# Projet IFT-712

Realised by :

 - Félix GAUCHER (CIP: gauf2611)
 - Adrien Verdier (CIP: vera2704)

Here you will find the code of our project.
In this file you can also find how to setup the development environment and how the project is structured.

## Setup

Create an isolated environment using your preferred solution 
(`venv`, `pipenv`,...) and install the required package: 
```
pip install -r requirements.txt
```

## Run the project

Dans cette partie, nous allons expliquer comment fonctionne notre programme et comment il peut être lancé simplement pour entraîner un modèle et vérifier son fonctionnement en effectuant une prédiction. Dans notre programme, nous prenons 80% des données de la base de données pour l'entraînement et 20% pour la validation. Pour lancer notre projet, il suffit de se placer dans le répertoire du projet et lancer la ligne de commande suivante : 

python3 run.py modele preprocessing grid_search

Tout d’abord, le paramètre modèle va représenter le modèle qu’on souhaite entraîner. On va avoir plusieurs possibilité : 
- PPV : méthode des plus proches voisins
- ADL : Analyse discriminante linéaire
- RFC : random forest
- SVC : méthode à noyau
- PCT : Perceptron
- RN : réseau de neuron
- DoubleSearch : Nous expliquerons plus tard son principe

	Ensuite, nous avons le paramètre preprocessing qui va indiquer le type de preprocessing que l’on va venir tenter d’effectuer sur nos données (nous expliquerons plus tard leurs principes) :
- None
- Scaled
- MinMax
- MaxAbs
- Quantile
- Gaussian
- Normalize

	Et enfin, grid_search est un paramètre binaire (0 ou 1) qui va déterminer si on effectue une grid_search sur le modèle ou non. Dans le cas de la double recherche, on effectue obligatoirement une grid search.


## Project structure
```
|   .gitignore
|   README.md
|   requirements.txt
|   run.py
|   
+---data
|   |   .gitkeep
|   |   
|   \---raw
|           .gitkeep
|           sample_submission.csv
|           test.csv
|           train.csv
|           
\---src
    |   .gitkeep
    |   two_step_prediction.py
    |   __init__.py
    |   
    +---data
    |       .gitkeep
    |       gestion_donnees.py
    |       
    +---features
    |       .gitkeep
    |       decoupage_donnees.py
    |       
    \---models
            .gitkeep
            Analyse_discriminante_lineaire.py
            noyau.py
            perceptron.py
            Plus_Proche_Voisin.py
            random_forest.py
            reseau_neurone.py
            __init__.py

```
    
