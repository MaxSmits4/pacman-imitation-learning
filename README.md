# Pacman - Imitation Learning

Projet réalisé dans le cadre du cours **INFO8006 - Introduction to Artificial Intelligence** (ULiège).

## Description

L'objectif est d'entraîner un réseau de neurones à imiter un joueur expert de Pacman. Le modèle apprend à partir de paires (état du jeu, action de l'expert) et prédit quelle action prendre dans chaque situation.

## Installation

```bash
pip install torch pandas
```

## Utilisation

**1. Entraîner le modèle :**
```bash
python train.py
```
Cela génère `pacman_model.pth`.

**2. Lancer le jeu :**
```bash
python run.py
```

## Structure

- `architecture.py` - Réseau de neurones (MLP)
- `data.py` - Extraction des features et dataset
- `train.py` - Pipeline d'entraînement
- `pacmanagent.py` - Agent qui utilise le modèle
- `run.py` - Lance une partie avec l'agent entraîné