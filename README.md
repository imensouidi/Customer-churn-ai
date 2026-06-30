# Prédiction du Churn Client avec le Machine Learning et le Deep Learning

## Présentation du projet

Ce projet a pour objectif de prédire le **churn client** (résiliation d'un abonnement) à l'aide de techniques de **Machine Learning** et de **Deep Learning**.

L'objectif est d'identifier les clients susceptibles de quitter l'entreprise afin de mettre en place des actions de fidélisation ciblées, d'améliorer la rétention des clients et de limiter les pertes de revenus.

Le projet suit l'ensemble du cycle de vie d'un projet de Data Science, depuis l'analyse exploratoire des données jusqu'au développement d'un tableau de bord interactif.

---

## Problématique métier

Le churn client représente un enjeu majeur pour les entreprises fonctionnant sur un modèle d'abonnement. Conserver un client existant est généralement moins coûteux que d'en acquérir un nouveau.

Ce projet vise à développer un modèle capable de prédire les clients présentant un risque élevé de résiliation afin d'aider les équipes métier à anticiper ce risque et à mettre en œuvre des actions de rétention adaptées.

Le problème est formulé comme une **classification binaire** :

- **0** → Le client reste fidèle
- **1** → Le client résilie son abonnement

---

## Structure du projet

```text
Customer-Churn-AI/
│
├── data/
│   └── customer_churn_business_dataset.csv
│
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── mlp.pkl
│   ├── preprocessor.pkl
│   ├── X_test_prepared.npy
│   └── y_test.npy
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Modeling.ipynb
│   └── 04_Evaluation.ipynb
│
├── dashboard/
│   └── app.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Jeu de données

Le projet s'appuie sur un **jeu de données synthétique** composé de **10 000 clients**.

Il contient des informations démographiques, comportementales, financières ainsi que des indicateurs de satisfaction client.

### Variables principales

### Profil client

- customer_id
- gender
- age
- country
- city
- customer_segment

### Abonnement

- contract_type
- signup_channel
- tenure_months

### Activité du client

- monthly_logins
- weekly_active_days
- avg_session_time
- features_used
- usage_growth_rate
- last_login_days_ago

### Informations financières

- monthly_fee
- total_revenue
- payment_method
- payment_failures
- discount_applied
- price_increase_last_3m

### Support client

- support_tickets
- avg_resolution_time
- complaint_type

### Satisfaction client

- csat_score
- nps_score
- survey_response

### Marketing

- email_open_rate
- marketing_click_rate
- referral_count

### Variable cible

- **0** → Client fidèle
- **1** → Client en churn

---

## Notebooks du projet

### 01 – Analyse exploratoire des données (EDA)

Cette étape permet de mieux comprendre le jeu de données grâce à :

- Présentation du jeu de données
- Statistiques descriptives
- Recherche des valeurs manquantes
- Détection des doublons
- Analyse de la distribution du churn
- Analyse univariée et multivariée
- Corrélations entre variables
- Test du Chi²
- Coefficient de V de Cramér
- Visualisations (boxplots, heatmaps...)

---

### 02 – Prétraitement des données

Cette étape prépare les données pour l'entraînement des modèles.

Principales étapes :

- Suppression des doublons
- Traitement des valeurs manquantes
- Séparation des variables numériques et catégorielles
- Standardisation des variables numériques
- Encodage One-Hot des variables catégorielles
- Découpage Train/Test
- Rééquilibrage des classes avec **Random Under Sampling (RUS)**

Après prétraitement :

- **51 variables d'entrée**
- **19 variables numériques**
- **32 variables catégorielles issues du One-Hot Encoding**

---

### 03 – Modélisation

Plusieurs modèles ont été entraînés et comparés.

#### Modèles de Machine Learning

- Régression Logistique
- Random Forest
- XGBoost

#### Modèle de Deep Learning

- Multi-Layer Perceptron (MLP)

Différentes stratégies de rééquilibrage des classes ont été évaluées.

Le modèle retenu est :

**Random Forest + Random Under Sampling (RUS)**

---

### 04 – Évaluation du modèle

Cette étape permet de valider les performances du modèle retenu.

#### Métriques d'évaluation

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Average Precision (PR-AUC)

#### Analyses réalisées

- Matrice de confusion
- Courbe ROC
- Courbe Precision-Recall
- Optimisation du seuil de décision
- Validation croisée (5-Fold Cross Validation)

#### Interprétabilité du modèle

- Feature Importance
- Permutation Importance
- SHAP Feature Importance
- SHAP Beeswarm
- SHAP Waterfall

Ces analyses permettent d'expliquer les prédictions du modèle aussi bien à un niveau global qu'à l'échelle d'un client.

---

## Tableau de bord interactif

Un tableau de bord interactif développé avec **Streamlit** permet d'utiliser le modèle de manière intuitive.

Le tableau de bord comprend plusieurs modules.

### Vue d'ensemble

- Indicateurs clés (KPI)
- Distribution du churn
- Analyse des segments clients
- Analyse des types de contrat
- Estimation de l'impact métier
- Estimation du retour sur investissement (ROI)

### Prédiction d'un client

L'utilisateur peut simuler le profil d'un client et obtenir :

- La probabilité de churn
- La classe prédite
- Le niveau de risque
- Les principaux facteurs de risque
- Des recommandations CRM
- Le revenu potentiellement à risque

### Évaluation des modèles

Comparaison des modèles selon :

- Les métriques d'évaluation
- Les courbes ROC
- Les courbes Precision-Recall
- Les matrices de confusion

### Analyse des variables

Visualisation de :

- Feature Importance
- Permutation Importance

afin de mieux comprendre les variables utilisées par le modèle.

---

## Technologies utilisées

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Scikit-learn
- XGBoost
- TensorFlow
- SHAP
- Streamlit
- Joblib
- Jupyter Notebook

---

## Installation

Cloner le dépôt :

```bash
git clone https://github.com/votre-utilisateur/customer-churn-ai.git
```

Se placer dans le dossier du projet :

```bash
cd customer-churn-ai
```

Créer un environnement virtuel :

```bash
python -m venv .venv
```

Activer l'environnement.

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Installer les dépendances :

```bash
pip install -r requirements.txt
```

---

## Exécution du projet

Exécuter les notebooks dans l'ordre suivant :

```text
01_EDA.ipynb
        ↓
02_Preprocessing.ipynb
        ↓
03_Modeling.ipynb
        ↓
04_Evaluation.ipynb
```

Lancer ensuite le tableau de bord :

```bash
streamlit run dashboard/app.py
```

---

## Résultats

L'étude comparative a montré que **Random Forest associé au Random Under Sampling (RUS)** offre le meilleur compromis entre :

- capacité de détection des churners (Recall),
- équilibre global des performances (F1-Score),
- qualité de discrimination (ROC-AUC),
- interprétabilité du modèle.

Ce projet illustre un workflow complet de Data Science comprenant :

- l'analyse exploratoire des données ;
- le prétraitement des données ;
- la comparaison de modèles de Machine Learning et de Deep Learning ;
- l'évaluation et l'interprétation des prédictions ;
- le développement d'un tableau de bord d'aide à la décision.

---

## Auteurs

- **Riad Boutria**
- **Imen Souidi**

---

## Licence

Ce projet a été réalisé dans un cadre pédagogique.
