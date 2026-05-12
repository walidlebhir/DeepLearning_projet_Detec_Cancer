# Modèle de classification des tumeurs cérébrales

## 1. Présentation générale
Ce projet utilise un modèle de classification d'images basé sur le transfert d'apprentissage. Le modèle principal est `EfficientNet-B0`, une architecture de réseau de neurones convolutionnel conçue pour offrir un bon compromis entre précision et efficacité.

Le but est de classer quatre catégories de cas IRM cérébrales :
- `glioma`
- `meningioma`
- `notumor`
- `pituitary`

Le pipeline comporte :
- prétraitement des images,
- entraînement d'une tête de classification adaptée,
- prédiction de la classe,
- génération de cartes d'attention Grad-CAM pour expliquer les décisions.

## 2. Pourquoi le transfert d'apprentissage ?
Le transfert d'apprentissage permet de réutiliser un réseau pré-entraîné sur une grande base d'images générales (comme ImageNet) puis de l'adapter à une tâche spécifique de classification médicale.

Avantages :
- meilleur démarrage que l'entraînement depuis zéro,
- besoin de moins de données annotées,
- convergence plus rapide,
- meilleure généralisation sur des jeux de données limités.

## 3. Architecture du modèle
### Base
- Base : `EfficientNet-B0`
- Utilisation : modèle pré-entraîné avec des poids de base (pas entraîné depuis zéro)

### Tête de classification ajoutée
Le modèle ajoute une nouvelle couche de sortie adaptée à 4 classes :
- `Dropout(0.3)`
- `Linear(1280, 256)`
- `ReLU()`
- `Dropout(0.3)`
- `Linear(256, 4)`

Cette tête permet de fine-tuner le modèle sur le problème spécifique des tumeurs cérébrales.

## 4. Types de données et organisation
Le projet utilise une structure de données issue de l'archive de classification :

- `Training/`
  - `glioma/`
  - `meningioma/`
  - `notumor/`
  - `pituitary/`

- `Testing/`
  - `glioma/`
  - `meningioma/`
  - `notumor/`
  - `pituitary/`

Chaque dossier contient des images d'IRM. Le modèle attend des images en couleur RGB, redimensionnées à `224x224`.

### Prétraitement
Pour l'entraînement et l'inférence, les transformations sont :
- redimensionnement en `224x224`,
- conversion en tenseur PyTorch,
- normalisation avec les moyennes et écarts-types ImageNet :
  - moyenne = `[0.485, 0.456, 0.406]`
  - écart type = `[0.229, 0.224, 0.225]`

## 5. Augmentation et robustesse
Le notebook utilise des augmentations pour améliorer la généralisation :
- rotation aléatoire,
- flips horizontaux et verticaux,
- affine randomisé,
- ColorJitter (variation de luminosité et contraste),
- RandomGrayscale,
- normalisation.

Ces transformations aident à rendre le modèle plus robuste aux variations d'IRM.

## 6. Entraînement du modèle
### Étapes principales
1. Charger les données d'entraînement et de validation.
2. Définir le modèle EfficientNet-B0 avec une tête de classification personnalisée.
3. Geler ou dégeler une partie des couches de la base selon le type de fine-tuning souhaité.
4. Choisir un optimiseur comme `AdamW`.
5. Utiliser un scheduler de taux d'apprentissage comme `CosineAnnealing`.
6. Calculer la perte et revenir en arrière (`backpropagation`).
7. Appliquer un `gradient clipping` si nécessaire.
8. Enregistrer le meilleur checkpoint.

### Objectif
- Maximiser l'exactitude globale,
- améliorer le rappel sur les classes critiques comme `glioma`.

## 7. Évaluation
L'évaluation du modèle inclut :
- matrice de confusion,
- classification report,
- scores ROC/AUC,
- courbes ROC,
- visualisation des prédictions.

## 8. Prédiction et API
Le service FastAPI exposé dans `src/main.py` et `src/Service/Detection_cancer.py` permet :
- recevoir une image uploadée,
- prédire la classe de la tumeur,
- renvoyer les probabilités de chaque classe,
- générer une image Grad-CAM encodée en base64.

### Endpoint principal
- `POST /detection/predict`

### Format de réponse
- `filename`
- `predicted_class`
- `confidence`
- `probabilities`
- `gradcam_image_base64`

## 9. Concept du modèle
### Ce que le modèle apprend
- la base EfficientNet extrait des caractéristiques visuelles profondes : textures, formes, contours,
- la tête de classification apprend à distinguer les quatre catégories de tumeurs à partir de ces caractéristiques.

### Pourquoi Grad-CAM ?
Grad-CAM fournit une explication visuelle de la zone de l'image sur laquelle le modèle s'est concentré pour prendre sa décision. Cela augmente la transparence, surtout en contexte médical.

## 10. Conseils pour l'entraînement
- utilisez un GPU si possible,
- veillez à un bon équilibrage des classes,
- surveillez l'overfitting via validation,
- conservez des checkpoints et des copies du modèle final.

## 11. Fichiers importants
- `src/Service/Detection_cancer.py` : service FastAPI et prédiction Grad-CAM,
- `src/main.py` : application FastAPI,
- `requirements.txt` : dépendances du projet,
- `DockerFile` et `docker-compose.yml` : conteneurisation.

## 12. Résumé
Ce projet met en œuvre un pipeline complet de classification de tumeurs cérébrales basé sur EfficientNet-B0 et le transfert d'apprentissage. Il combine entraînement d'un classifieur spécialisé avec des outils d'explication visuelle, et fournit une API prête à l'emploi pour déployer le modèle.
