Project Name
==============================

This project is a starting Pack for MLOps projects based on the subject "London Fire Brigade Response Time". It's not perfect so feel free to make some modifications on it.

Project Organization
------------

    
    ├── JUIN23_CONTINU_MLOPS_POMPIERS    
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data
    │   │    ├── import_raw_data     <- Raw data import from MySQL database.
    │   │    └── make_dataset        <- Preparing variables and data transformation to create a Working Datase.
    │   │
    │   ├── models_training
        │     └── model               <- Training and serialization of the model.
    │   │
    │   ├── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   │   └── visualize.py
    │   └── config         <- Describe the parameters used in train_model.py and predict_model.py
    ├── LICENSE
    ├── README.md
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
--------

<p><small>Project based on the <a target="_blank" href="https://data.london.gov.uk/dataset/london-fire-brigade-incident-records">LFB's data for incidents </a> and <a target="_blank" href="https://data.london.gov.uk/dataset/london-fire-brigade-incident-records">LFB's data for mobilisations </a>.</small></p>
