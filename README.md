# Système Intelligent de Gestion de Présence avec Reconnaissance Faciale

## 📋 Table des Matières
- [Introduction](#introduction)
- [Fonctionnalités](#fonctionnalités)
- [Architecture du Système](#architecture-du-système)
- [Prérequis et Installation](#prérequis-et-installation)
- [Structure du Projet](#structure-du-projet)
- [Guide d'Utilisation](#guide-dutilisation)
- [Développement et Personnalisation](#développement-et-personnalisation)
- [Dépannage](#dépannage)
- [Contributions](#contributions)
- [Licence](#licence)

---

## 🎯 Introduction

Ce projet propose une solution complète d'**Industrie 4.0** pour automatiser la gestion des présences dans un environnement éducatif grâce à la **reconnaissance faciale**. Le système combine intelligence artificielle, IoT et automatisation pour remplacer les méthodes traditionnelles de prise de présence.

### 🎓 Contexte
Les méthodes traditionnelles (feuilles papier, appels nominaux) sont :
- ⏱️ Chronophages et peu fiables
- 📝 Sujettes aux falsifications et oublis
- 🔄 Inadaptées à l'ère numérique

### 💡 Solution Proposée
Un système automatisé qui :
- ✅ Identifie les étudiants par reconnaissance faciale
- 📊 Génère des rapports automatiques
- 📧 Envoie les résultats par email
- 🖥️ Fournit une interface de gestion

---

## ✨ Fonctionnalités

### 🎭 Reconnaissance Faciale
- **Détection en temps réel** via webcam
- **Identification précise** avec modèle FaceNet
- **Gestion des inconnus** avec capture automatique
- **Seuil de confiance** configurable

### 📈 Gestion des Présences
- **Enregistrement automatique** des présences/absences
- **Génération de rapports** Excel détaillés
- **Statistiques visuelles** (graphiques et camemberts)
- **Historique temporel** avec horodatage

### 🔧 Interface Utilisateur
- **Dashboard enseignant** avec Streamlit
- **Configuration flexible** des paramètres
- **Visualisation en temps réel**
- **Contrôle manuel** du système

### 📧 Communication
- **Envoi automatique** d'emails avec pièces jointes
- **Rapports complets** incluant captures d'écran
- **Notifications** de statut système

---

## 🏗️ Architecture du Système

### 📊 Schéma Global
```
Camera → Traitement Image → Extraction Features → Classification → Résultats
                                   ↓
                          Dashboard Enseignant ←→ Base de Données
                                   ↓
                             Envoi Email + Rapports
```

### 🔄 Pipeline de Traitement
1. **Acquisition** : Capture vidéo via webcam
2. **Prétraitement** : Détection et alignement (MTCNN)
3. **Extraction** : Embeddings 512D (FaceNet)
4. **Classification** : Identification (SVM)
5. **Post-traitement** : Génération rapports + notifications

---

## ⚙️ Prérequis et Installation

### 🛠️ Matériel Requis
- **Raspberry Pi 4** (4GB RAM recommandé)
- **Webcam USB** (1080p minimum)
- **Écran LCD** 16x2 avec interface I2C
- **Carte SD** 16GB+ (Classe 10)
- **Alimentation** 5V/3A

### 📦 Logiciels et Bibliothèques

#### Système d'Exploitation
```bash
# Installer Raspberry Pi OS via Raspberry Pi Imager
sudo apt update && sudo apt upgrade -y
```

#### Dépendances Python
```bash
# Installation des packages système
sudo apt install python3-pip python3-opencv -y

# Installation des bibliothèques Python
pip install opencv-python
pip install facenet-pytorch
pip install torch torchvision
pip install pandas scikit-learn joblib
pip install streamlit RPLCD smtplib
pip install pillow matplotlib seaborn
```

#### Configuration Matérielle
```bash
# Activer la caméra
sudo raspi-config
# → Interface Options → Camera → Enable

# Activer I2C pour l'écran LCD
sudo raspi-config
# → Interface Options → I2C → Enable
```

---

## 📁 Structure du Projet

```
Projet_Reconnaissance_Faciale/
│
├── 📊 Données/
│   ├── eleves/                          # Base de données des visages
│   │   ├── nom_prenom1/
│   │   │   ├── image1.jpg
│   │   │   └── image2.jpg
│   │   └── nom_prenom2/
│   │       └── ...
│   ├── inconnus/                        # Visages non reconnus
│   └── modeles/                         # Modèles entraînés
│       ├── face_classifier.pkl
│       └── label_encoder.npy
│
├── 🔧 Code Source/
│   ├── main_raspberry.py               # Script principal Raspberry
│   ├── Construction_modele.ipynb       # Notebook d'entraînement
│   ├── interface_enseignant.py         # Dashboard Streamlit
│   └── utils/
│       ├── preprocessing.py
│       ├── email_sender.py
│       └── lcd_controller.py
│
├── 📋 Documentation/
│   ├── README.md
│   ├── guide_installation.pdf
│   └── manuel_utilisation.pdf
│
└── 📈 Résultats/
    ├── liste_presence.xlsx             # Fichier généré automatiquement
    └── logs_detection.csv              # Historique des détections
```

---

## 🚀 Guide d'Utilisation

### 1. 🏗️ Construction du Modèle

#### Préparation des Données
```python
# Structure des dossiers obligatoire :
eleves/
├── nom_prenom1/
│   ├── photo1.jpg
│   └── photo2.jpg
├── nom_prenom2/
│   └── ...
```

#### Entraînement du Modèle
```bash
# Exécuter le notebook d'entraînement
jupyter notebook Construction_modele.ipynb

# Ou exécuter le script Python
python train_model.py
```

### 2. 🎯 Déploiement sur Raspberry Pi

#### Configuration Initiale
```bash
# Copier les fichiers sur le Raspberry Pi
scp -r Projet_Reconnaissance_Faciale/ pi@raspberrypi.local:~/

# Tester la caméra
python3 -c "import cv2; print(cv2.__version__)"
```

#### Lancement du Système
```bash
# Mode automatique
python3 main_raspberry.py

# Mode manuel avec interface
streamlit run interface_enseignant.py
```

### 3. ⚙️ Configuration des Paramètres

#### Paramètres de Détection
- **Durée de détection** : 3-30 minutes
- **Seuil de confiance** : 0.7 (recommandé)
- **Taille minimale visage** : 200 pixels
- **Temps de stabilisation** : 7 secondes

#### Configuration Email
```python
EMAIL_EXPEDITEUR = "votre.email@gmail.com"
MOT_DE_PASSE = "mot_de_passe_application"
DESTINATAIRE = "enseignant@etablissement.edu"
```

---

## 🔧 Développement et Personnalisation

### 🎨 Personnalisation du Modèle

#### Ajout d'Étudiants
```python
# Structure requise pour nouveaux étudiants
nouvel_etudiant/
├── photo_face_1.jpg
├── photo_face_2.jpg
└── photo_face_3.jpg  # Minimum 3 photos recommandé
```

#### Réentraînement du Modèle
```python
# Régénérer le dataset
df = create_dataset("eleves/")

# Réentraîner le classifieur
classifier.fit(embeddings, labels)
joblib.dump(classifier, 'face_classifier.pkl')
```

### 🔌 Extension des Fonctionnalités

#### Ajout de Nouveaux Capteurs
```python
# Exemple : Intégration capteur température
def lire_temperature():
    # Code pour lire capteur DHT22
    return temperature
```

#### Intégration Base de Données
```python
# Exemple : Sauvegarde PostgreSQL
import psycopg2
conn = psycopg2.connect("votre_connection_string")
```

---

## 🐛 Dépannage

### ❌ Problèmes Courants

#### Caméra Non Détectée
```bash
# Vérifier la détection
vcgencmd get_camera
# Expected: supported=1 detected=1

# Redémarrer le service
sudo systemctl restart raspberrypi_camera
```

#### Erreur Mémoire Insuffisante
```bash
# Augmenter le swap
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Modifier CONF_SWAPSIZE=1024
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

#### Problèmes I2C (LCD)
```bash
# Vérifier la détection
i2cdetect -y 1
# Vérifier l'adresse (généralement 0x27)

# Vérifier les connexions
sudo apt install i2c-tools
```

### 📊 Debugging Avancé

#### Logs Détaillés
```python
# Activer les logs détaillés
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### Test Composant par Composant
```python
# Test caméra
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Erreur')"

# Test LCD
python3 -c "from RPLCD.i2c import CharLCD; lcd = CharLCD('PCF8574', 0x27); lcd.write_string('Test OK')"
```

---

## 🤝 Contributions

### 🎯 Améliorations Futures
- [ ] Intégration reconnaissance masques faciaux
- [ ] Analyse des émotions
- [ ] Interface mobile
- [ ] Synchronisation cloud
- [ ] Alertes en temps réel

### 📝 Guide de Contribution
1. 🍴 Fork le projet
2. 🌿 Créer une branche feature (`git checkout -b feature/AmazingFeature`)
3. 💾 Commit les changements (`git commit -m 'Add AmazingFeature'`)
4. 📤 Push la branche (`git push origin feature/AmazingFeature`)
5. 🔀 Ouvrir une Pull Request

---


## 📞 Support

Pour toute question ou problème :
- 📧 Email : Oussamafahim2017@gmail.com
- Télephone : +212645468306

---

## 🙏 Remerciements

- **Encadrant** : Pr. Mohamed Hosni
- **Équipe de Développement** :
  - Salma Bourkiba
  - Hasna Jhabli  
  - Meryem Filali Ansari
  - Ouiame Bellaknich
  - Mariam Lakhsassi
  - Fatima El Fadili
  - Salma Oulkiass

- **Contributeurs** : Tous les testeurs et personnes ayant participé à l'amélioration du système.

---

**⭐ Si ce projet vous a été utile, n'hésitez pas à lui donner une étoile sur GitHub !**
