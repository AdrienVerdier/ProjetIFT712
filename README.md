# ProjetIFT712

*FelixGaucher - AdrienVerdier, 06/10/2020*

Projet de session du cours IFT 712

## Setup

Create an isolated environment using your preferred solution 
(`venv`, `pipenv`,...) and install the required package: 
```
pip install -r requirements.txt
```


## Project structure
```
├── requirements.txt         <- The requirements file for reproducing the analysis 
|                               environment. 

├── README.md                <- The top-level README
├── run.py                   <- Script with option for running the final analysis.
├── data
|   ├── interim              <- Intermediate data that has been transformed.
│   ├── processed            <- The final, canonical data sets for modeling.
│   └── raw                  <- The original, immutable data dump.
├── notebooks                <- Jupyter notebooks.
├── output             
|   ├── models               <- Serialized models, predictions, model summaries.
|   └── visualization        <- Graphics created during analysis.
└── src                      <- Source code for this project.
    ├── __init__.py          <- Makes this a python module.
    ├── data                 <- Scripts to download or generate data.
    |   └── make_dataset.py  
    ├── features             <- Scripts to turn raw data into features for modeling.
    |   └── build_features.py  
    ├── models               <- Scripts used to generate models and inference results.
    └── visualization        <- Scripts to generate graphics.
        └── visualize.py
```
    
