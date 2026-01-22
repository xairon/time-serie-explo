# Guide de Déploiement - Time Series Explo

Ce guide explique comment déployer facilement le projet Streamlit sur différentes architectures.

## 🚀 Déploiement Rapide

### Étape 1 : Prérequis

- Python 3.10, 3.11 ou 3.12
- pip (généralement inclus avec Python)
- Git (pour cloner le repository)

### Étape 2 : Installation Automatique

Le script `setup_env.py` gère automatiquement toute l'installation :

```bash
# Mode interactif (recommandé)
python setup_env.py

# Ou spécifiez directement l'architecture :
python setup_env.py --device cpu   # CPU standard
python setup_env.py --device cuda  # NVIDIA GPU
python setup_env.py --device xpu   # Intel Arc GPU
```

Le script va :
1. ✅ Créer un environnement virtuel
2. ✅ Installer PyTorch avec l'index-url approprié
3. ✅ Installer toutes les dépendances de base
4. ✅ Vérifier l'installation

### Étape 3 : Vérification

```bash
python verify_installation.py --venv venv --device cpu
```

### Étape 4 : Lancement

```bash
# Activer l'environnement
# Windows:
venv\Scripts\activate

# Linux/Mac:
source venv/bin/activate

# Lancer l'application
python run_app.py
```

L'application sera accessible sur `http://localhost:8501`

---

## 📋 Déploiement par Architecture

### CPU (Recommandé pour débuter)

```bash
python setup_env.py --device cpu
python verify_installation.py --venv venv --device cpu
python run_app.py
```

**Avantages** :
- ✅ Fonctionne partout
- ✅ Installation simple
- ✅ Pas de dépendances matérielles

**Inconvénients** :
- ⚠️ Entraînement plus lent

### NVIDIA CUDA (GPU NVIDIA)

**Prérequis** :
- GPU NVIDIA compatible
- Drivers NVIDIA installés
- CUDA 11.8 (ou compatible)

```bash
python setup_env.py --device cuda
python verify_installation.py --venv venv --device cuda
python run_app.py
```

**Vérification CUDA** :
```python
import torch
print(torch.cuda.is_available())  # Doit retourner True
print(torch.cuda.get_device_name(0))  # Nom du GPU
```

### Intel XPU (Intel Arc GPU)

**Prérequis** :
- GPU Intel Arc (A770, A750, etc.)
- Intel oneAPI Base Toolkit installé
- Drivers Intel Arc installés

```bash
python setup_env.py --device xpu --venv venv_arc
python verify_installation.py --venv venv_arc --device xpu
python run_app.py
```

**Vérification XPU** :
```python
import torch
print(hasattr(torch, 'xpu') and torch.xpu.is_available())  # Doit retourner True
```

---

## 🔧 Installation Manuelle (Alternative)

Si vous préférez installer manuellement :

### 1. Créer l'environnement virtuel

```bash
python -m venv venv

# Activer
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate
```

### 2. Installer PyTorch

**CPU** :
```bash
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cpu
```

**CUDA** :
```bash
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu118
```

**XPU** :
```bash
pip install torch==2.10.0+xpu torchaudio==2.10.0+xpu torchvision==0.25.0+xpu --index-url https://download.pytorch.org/whl/test/xpu
```

### 3. Installer les dépendances de base

```bash
pip install -r requirements/base.txt
```

---

## 🐳 Déploiement Docker

### Build et Run

```bash
docker-compose up -d --build
```

### Voir les logs

```bash
docker-compose logs -f
```

### Arrêter

```bash
docker-compose down
```

L'application sera accessible sur le port **49500** (configurable dans `docker-compose.yml`).

---

## ✅ Checklist de Déploiement

- [ ] Python 3.10+ installé
- [ ] Repository cloné
- [ ] `setup_env.py` exécuté avec succès
- [ ] `verify_installation.py` passe tous les tests
- [ ] Environnement virtuel activé
- [ ] Application démarre sans erreur
- [ ] Interface accessible sur `http://localhost:8501`

---

## 🐛 Résolution de Problèmes

### Problème : "pip not found"

**Solution** : Vérifiez que Python est dans votre PATH et que pip est installé :
```bash
python -m ensurepip --upgrade
```

### Problème : "PyTorch installation fails"

**Solution** :
1. Vérifiez votre connexion internet
2. Essayez avec `--no-cache-dir` :
   ```bash
   pip install --no-cache-dir torch torchaudio torchvision --index-url <url>
   ```
3. Vérifiez que vous utilisez la bonne version de Python (3.10-3.12)

### Problème : "CUDA not available" (avec --device cuda)

**Solution** :
1. Vérifiez que les drivers NVIDIA sont installés : `nvidia-smi`
2. Vérifiez que CUDA 11.8 est installé
3. PyTorch peut fonctionner en mode CPU même si installé avec CUDA

### Problème : "XPU not available" (avec --device xpu)

**Solution** :
1. Vérifiez que Intel oneAPI Base Toolkit est installé
2. Vérifiez que les drivers Intel Arc sont à jour
3. PyTorch peut fonctionner en mode CPU même si installé avec XPU

### Problème : "Streamlit fails to start"

**Solution** :
1. Vérifiez que l'environnement virtuel est activé
2. Vérifiez l'installation : `python verify_installation.py`
3. Essayez de lancer directement : `streamlit run dashboard/training/Home.py`
4. Vérifiez qu'aucun autre processus n'utilise le port 8501

### Problème : "Module not found"

**Solution** :
1. Vérifiez que l'environnement virtuel est activé
2. Réinstallez les dépendances : `pip install -r requirements/base.txt`
3. Vérifiez que vous êtes dans le répertoire racine du projet

---

## 📝 Notes Importantes

1. **Environnements multiples** : Vous pouvez créer plusieurs environnements pour différentes architectures :
   ```bash
   python setup_env.py --device cpu --venv venv_cpu
   python setup_env.py --device cuda --venv venv_cuda
   python setup_env.py --device xpu --venv venv_xpu
   ```

2. **Port personnalisé** : Changez le port avec :
   ```bash
   python run_app.py --port 8502
   ```

3. **Mise à jour des dépendances** : Pour mettre à jour :
   ```bash
   pip install --upgrade -r requirements/base.txt
   ```

4. **Réinstallation complète** : Pour tout réinstaller :
   ```bash
   # Supprimer l'ancien venv
   rm -rf venv  # Linux/Mac
   rmdir /s venv  # Windows
   
   # Recréer
   python setup_env.py --device cpu
   ```

---

## 🎯 Prochaines Étapes

Une fois l'application lancée :

1. **Préparer les données** : Allez dans "Dataset Preparation"
2. **Entraîner un modèle** : Allez dans "Train Models"
3. **Faire des prédictions** : Allez dans "Forecasting"

Consultez le [README.md](README.md) pour plus d'informations sur l'utilisation de l'application.
