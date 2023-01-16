# MLBIO

Le diagnostique efficace du covid aurait pu être un outil efficace pour limiter la transmission du coronavirus si nos politiques publiques étaient efficaces et avaient implémenté la solution test-trace et isolate. 

Malheureusement les tests covid nécessitent de se déplacer dans des établissements pour faire le test et demandent du temps. 

Le but de ce projet est de développe un modèle de classification de son capable de distinguer une toux liée au covid d'une toux bénine à l'aide de l'enregistrement de la toux d'un patient.

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

Pour prédire si une toux est significative d'un cas COVID positive ou négatif à partir d'un fichier audio, il suffit de lancer la commande suivante:

```python
python predict.py <audio_path>
```

## Observations

Le modèle de classification `covid_cough_classifier.h5` a été généré dans le notebook `covid_audio_classification.ipynb`. Dans ce notebook, on a utilisé la librairie `librosa` pour extraire les mel-spectrograms des fichiers audio du dataset Coswara-Data.

On a ensuite utilisé la librairie `keras` pour construire un modèle de classification à couches convolutives pour prédire si une toux est significative d'un cas COVID-19 ou non. Cependant, le modèle n'a pas été entrainé sur un dataset de taille suffisante pour prédire avec une grande précision, et un overfitting est présent.
Le modèle réussi quand même à prédire avec un taux de 70% de précision si une toux est significative d'un cas COVID-19 ou non sur un dataset de test.

L'overfitting peut être à la fois dû à la non-présence significative d'élément distinctif sur les fichiers audio des toux permettant la prédiction de COVID-19 dans le dataset Coswara-Data, et à la taille du dataset utilisé pour l'entrainement du modèle, qui est de la forme suivante:

- Train data : negative (1089) | positive (495)
- Validation data : negative (273) | positive (124)
- Test data : negative (341) | positive (155)

## Sources

- [Coswara-Data](https://github.com/iiscleap/Coswara-Data)
- [Covid-19 Cough classification using machine learning](https://arxiv.org/pdf/2012.01926.pdf)