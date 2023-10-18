![Logo London Fire Brigade](https://upload.wikimedia.org/wikipedia/en/thumb/9/92/London_Fire_Brigade_logo.svg/1200px-London_Fire_Brigade_logo.svg.png)


London Fire Brigade Response Time
==============================

This is a MLOps project based on the subject "London Fire Brigade Response Time".
The objective of this project is to analyze and estimate the response times and mobilization of the London Fire Brigade. The London Fire Brigade is the busiest fire and rescue service in the United Kingdom and one of the largest firefighting and rescue organizations in the world.

Project Organization
------------

    ├── JUIN23_CONTINU_MLOPS_POMPIERS    
    │
    ├── .github\workflows
    │     └── python-app.yml
    ├── docker-compose.yml
    ├── LICENSE
    ├── models
    ├── notebooks
    │   ├── 1.0-wm-data-exploration-and-testing-models.ipynb
    │   ├── 2.0-wm-data-exploration-and-testing-models.ipynb
    │   ├── 2.1-wm-testing-models-features-selection.ipynb
    │   └── 2.2-wm-random-forest-features-selection.ipynb
    ├── README.md
    ├── references
    │   ├── Cahier des charges LFB V2.docx
    │   ├── Metadata
    │   │   ├── Incidents Metadata.xlsx
    │   │   └── Mobilisations Metadata.xlsx
    │   └── Prédiction du temps de réponse des pompiers.docx
    ├── requirements.txt
    ├── setup.py
    └── src
        ├── api_admin
        │   ├── api
        │   │   ├── __init__.py
        │   │   ├── schema.py
        │   │   └── users.py
        │   ├── api_admin.py
        │   ├── data
        │   │   ├── import_raw_data.py
        │   │   ├── __init__.py
        │   │   ├── make_dataset.py
        │   │   └── __pycache__
        │   │       ├── import_raw_data.cpython-310.pyc
        │   │       ├── __init__.cpython-310.pyc
        │   │       └── make_dataset.cpython-310.pyc
        │   ├── Dockerfile
        │   ├── models_training
        │   │   ├── __init__.py
        │   │   ├── model.py
        │   ├── test_api_admin.py
        │   └── tests
        │       ├── __init__.py
        │       ├── test_import_raw_data.py
        │       └── test_model.py
        └── api_user
            ├── api
            │   ├── __init__.py
            │   ├── schema.py
            │   └── users.py
            ├── api_user.py
            ├── data
            │   ├── import_raw_data.py
            │   ├── __init__.py
            │   └── make_dataset.py
            ├── Dockerfile
            ├── models_training
            │   ├── __init__.py
            │   └── model.py
            └── test_api_user.py



--------

Context
------------

The London Fire Brigade (LFB) is the largest fire service in the world with 103 stations and over 5000 professional firefighters. The Brigade covers the 13 boroughs of London, which is home to 8 million residents. The Brigade responds to between 100,000 and 130,000 emergency calls each year and operates in a territory of 1587 square kilometers.

In cases of life-threatening emergencies, every lost minute diminishes the chances of survival or increases the risk of lasting consequences. Regarding fires, a famous saying goes: the first minute, a fire can be extinguished with a glass of water, the second with a bucket, and the third with a tanker! It is therefore crucial for firefighters to reach the site as quickly as possible.

In this context, it would be beneficial for the brigade, especially for the call center, to anticipate and predict the intervention time of firefighters following a call. This would help reassure those calling for help, better optimize the logistics of emergency services, or challenge the organization of departure and routes to further reduce the response time to emergencies

------------


Application Operation
------------

**APIs :**
The application consists of two APIs built with FastAPI:
- API User :
The User API is available on port 8001.
It allows the user to 
    -  get a prediction of the time between the moment of the emergency call and the arrival of the first firefighting forces at the incident scene. 

- API Admin :
The Admin API is available on port 8002. 
It allows the admin to
    - consult the database structure
    - get a sample of data from the database
    - train a model
    - get metrics for a model

**Amazon AWS :**
The application uses Amazon AWS cloud services, including : 
- RDS : For the MySQL database
- S3 : To store various models, metrics, label encoders fitted to training data, and Min/Max scalers fitted to training data.

**Airflow :**
The admin has access to the Airflow interface, where DAGs allow regular model evaluation and training on new data.

**Streamlit :**
A Streamlit interface has been created for the user to facilitate testing of the application.

------------

Project Diagram
------------

insérer ici l'image du schéma

------------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
