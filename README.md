London Fire Brigade Response Time
==============================

This project is a starting Pack for MLOps projects based on the subject "London Fire Brigade Response Time". It's not perfect so feel free to make some modifications on it.

Project Organization
------------

    
    ├── JUIN23_CONTINU_MLOPS_POMPIERS    
    │
    ├── .github\workflows
    │     └── python-app.yml
    │
    ├── models         
    │     ├── label_encoder     <- Raw data import from MySQL database.
    │     └── model_lgb         <- Preparing variables and data transformation to create a Working Datase.
    
    
    
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    
    │
    
    │
    ├── src                
    │   ├── api
    │   │    ├── schema              <- Class for api.
    │   │    └── users               <- Database for users and authentication.
    │   │
    │   ├── data
    │   │    ├── import_raw_data     <- Raw data import from MySQL database.
    │   │    └── make_dataset        <- Preparing variables and data transformation to create a Working Datase.
    │   │
    │   ├── models_training
        │     └── model              <- Training and serialization of the model.
    │   │
    │   └── test         
    │         ├── test_import-raw_data <- Unit test for import_raw_data.
    │         └── test_model           <- Unit test for model.
    │
    ├── LICENSE
    ├── README.md
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    ├── Setup              <- Creating and distributing Python packages"
    │ 
--------

<p><small>Project based on the <a target="_blank" href="https://data.london.gov.uk/dataset/london-fire-brigade-incident-records">LFB's data for incidents </a> and <a target="_blank" href="https://data.london.gov.uk/dataset/london-fire-brigade-incident-records">LFB's data for mobilisations </a>.</small></p>
