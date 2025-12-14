# 🦠 Classification des Radiographies Pulmonaires - COVID-19

Ce projet utilise l'apprentissage profond pour classer les radiographies pulmonaires en deux catégories : **COVID-19** et **Non-COVID-19**. Le modèle est construit avec **TensorFlow/Keras** et entraîné sur un jeu de données d'images médicales.

---

## 📁 Structure du projet

```
covid_19.ipynb        # Notebook principal contenant tout le code
data/
  └── xray_dataset_covid19/
      ├── train/
      └── test/
```

---

## 🧠 Étapes principales du notebook

1. **Importation des bibliothèques**

   * tensorflow, keras
   * numpy, matplotlib
   * sklearn pour l'évaluation

2. **Prétraitement des données**

   * Redimensionnement des images à 128x128 pixels
   * Normalisation des pixels entre 0 et 1
   * Division des données :

     * 80% pour l'entraînement
     * 20% pour la validation
     * 10% pour le test

3. **Chargement et préparation des datasets**

   * Utilisation de `tf.keras.utils.image_dataset_from_directory`
   * Mélange et partitionnement des données
   * Optimisation avec `.cache()` et `.prefetch()`

4. **Construction du modèle**

   * Architecture CNN (non détaillée dans l'extrait fourni)
   * Compilation avec un optimiseur et une fonction de perte adaptée

5. **Entraînement**

   * Entraînement sur `train_ds`
   * Validation sur `val_ds`
   * Suivi des métriques (accuracy, perte)

6. **Évaluation**

   * Performance sur le jeu de test
   * Rapport de classification avec `sklearn.metrics.classification_report`

---

## 📊 Résultats

Les performances du modèle sont évaluées à l'aide :

* De la précision (**accuracy**)
* De la perte (**loss**)
* Du rapport de classification (**precision, recall, f1-score**)

---

## 🛠️ Technologies utilisées

* Python 3
* TensorFlow / Keras
* NumPy, Matplotlib
* Scikit-learn

---

## 📦 Installation et exécution

1. Clonez le dépôt :

```bash
git clone <url-du-repo>
cd <dossier-du-projet>
```

2. Installez les dépendances :

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

3. Placez le jeu de données dans `data/xray_dataset_covid19/`

4. Exécutez le notebook :

```bash
jupyter notebook covid_19.ipynb
```
