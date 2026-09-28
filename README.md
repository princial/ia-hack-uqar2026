# 🐋 IA'Hack 2026 (Rimouski) — Défi #1 : Bioacoustique Marine

Système d'intelligence artificielle pour la détection et la classification automatique de vocalisations de mammifères marins dans le cadre du défi **IA'Hack 2026 (Rimouski)**.

---

## 📋 Présentation du Projet

Ce projet aborde les défis de la surveillance acoustique sous-marine passive (PAM - *Passive Acoustic Monitoring*) :
1. **Partie #1 — Classification d'espèces marines** : Identification de l'espèce marine émettrice à partir d'enregistrements audio courts (Béluga, Rorqual commun, Baleine à bosse, Cachalot, Dauphin à flancs blancs ou Bruit ambiant).
2. **Partie #2 — Détection d'événements sonores dans de longues séquences** : Détection temporelle continue (timestamps de début et de fin) de vocalisations et estimation des probabilités de présence.
3. **Application Démo Interactive (`app_demo.py`)** : Interface Streamlit moderne permettant de tester le modèle, visualiser les spectrogrammes et comparer les prédictions aux annotations de vérité terrain.

---

## 🐬 Espèces Cibles

| Code Espèce | Nom Français | Nom Scientifique / Commun |
| :--- | :--- | :--- |
| `Beluga_WhiteWhale` | Béluga | *Delphinapterus leucas* |
| `Fin_FinbackWhale` | Rorqual commun | *Balaenoptera physalus* |
| `HumpbackWhale` | Baleine à bosse | *Megaptera novaeangliae* |
| `SpermWhale` | Cachalot | *Physeter macrocephalus* |
| `White_sidedDolphin` | Dauphin à flancs blancs | *Lagenorhynchus acutus* |
| `Noise` | Bruit marin / Ambiance | Fond sonore marin |

---

## 🛠️ Architecture & Pipeline Technique

- **Prétraitement Audio** : Rééchantillonnage standardisé (16 kHz), normalisation RMS, fenêtrage glissant avec chevauchement (*overlapping windows*).
- **Extraction de Caractéristiques Acoustiques** :
  - Coefficients cepstraux sur échelle de Mel (MFCCs) + dérivées delta & delta-delta.
  - Centroïde spectral, bande passante spectrale, rolloff spectral, flux spectral.
  - Statistiques descriptives (moyenne, écart-type, quantiles, asymétrie).
- **Modélisation & Inférence** :
  - Modèle entraîné et sérialisé dans `best_model.joblib`.
  - Optimisation des hyperparamètres de détection temporelle documentée dans `part2_best_params.json` (seuil de probabilité, tolérance de fusion des segments, etc.).

---

## 📁 Structure du Répertoire

```text
├── app_demo.py                 # Application Streamlit interactive de démonstration
├── run.py                      # Pipeline complet d'entraînement, validation & détection
├── best_model.joblib           # Modèle de classification pré-entraîné
├── part2_best_params.json      # Hyperparamètres optimaux pour la Partie #2
├── requirements.txt            # Dépendances Python du projet
├── .gitignore                  # Fichiers et dossiers exclus du suivi Git (env, cache...)
└── Ressources du défi #1/      # Données du défi
    ├── IA’Hack 2026 ... pdf    # Cahier des charges et description du défi
    ├── Partie #1/              # Base d'apprentissage et de test (clips courts par espèce)
    └── Partie #2/              # Séquences audio longues et annotations associées
```

---

## 🚀 Installation & Démarrage

### 1. Cloner le projet

```bash
git clone https://github.com/<votre-nom-d-utilisateur>/<nom-du-depot>.git
cd <nom-du-depot>
```

### 2. Créer et activer un environnement virtuel

Sous Windows (PowerShell) :
```powershell
python -m venv env
.\env\Scripts\Activate.ps1
```

Sous Linux / macOS :
```bash
python3 -m venv env
source env/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## 🖥️ Utilisation

### Lancer l'interface web interactive

```bash
streamlit run app_demo.py
```
L'application s'ouvrira automatiquement dans votre navigateur (par défaut sur `http://localhost:8501`). Vous pourrez y charger des fichiers audio `.wav`, visualiser les spectrogrammes et lancer les détections.

### Lancer le pipeline complet via le terminal

```bash
python run.py --help
```

---

## 📦 Dépendances Principales

- `numpy` & `pandas` : Traitement et manipulation des données
- `scikit-learn` : Modèles de classification et métriques d'évaluation
- `librosa` & `soundfile` : Traitement du signal audio et extraction acoustique
- `torch` & `torchaudio` : Outils complémentaires d'analyse audio
- `streamlit` : Interface utilisateur web interactive
- `matplotlib` & `seaborn` : Visualisations et spectrogrammes
