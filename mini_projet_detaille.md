# Mini-Projet : Transformers et Systèmes RAG
**Filière :** Ingénierie des Données (Deuxième Année)  
**Établissement :** École Nationale des Sciences Appliquées - Al Hoceima (ENSAH)

---

## 🎯 Objectif Général
Ce mini-projet vise à mettre en pratique les concepts avancés en Traitement du Langage Naturel (NLP). Le projet est divisé en deux parties complémentaires :
1. La prise en main des modèles **Transformers** via l'écosystème HuggingFace.
2. La conception, l'analyse et l'implémentation d'un système **RAG (Retrieval-Augmented Generation)** pour surpasser les limites des modèles de langage classiques (LLMs).

---

## 📌 Partie 1 : Exploration des Transformers et Tâches NLP

### 1. Objectif
Se familiariser avec l'utilisation pratique des architectures de type Transformer dans diverses tâches classiques de NLP. Cette partie s'appuie principalement sur la bibliothèque `transformers` de HuggingFace.

### 2. Travail Demandé
*   **Maîtrise de l'écosystème HuggingFace :**
    *   Chargement et utilisation de modèles pré-entraînés depuis le *Model Hub*.
    *   Manipulation des **tokenizers** (encodage et décodage du texte).
    *   Utilisation des **pipelines** simplifiés pour des cas d'usage rapides.
*   **Implémentation Pratique des tâches NLP :**
    *   **Classification de Texte** (ex. catégorisation thématique).
    *   **Analyse de Sentiment** (détection de la polarité positive/négative).
    *   **Question Answering (QA)** (extraction de réponses à partir d'un contexte fourni).
    *   **Résumé Automatique (Summarization)** (génération de synthèses concises pour de longs textes).

---

## 🧠 Partie 2 : Systèmes RAG (Retrieval-Augmented Generation)

### 1. Objectif
Comprendre le fonctionnement et implémenter de A à Z un pipeline RAG. L'idée est de pallier le manque de connaissances externes (ou les hallucinations) des LLMs en intégrant une étape de recherche d'informations (Retrieval) avant la génération (Generation).

### 2. Travail Demandé, divisé en 4 étapes majeures :

#### Étape 2.1 : État de l'art et Recherche (Théorie)
Présenter les fondements théoriques et conceptuels :
*   **Définition** d'un système RAG.
*   **Limites des LLMs** classiques (bases de connaissances figées, hallucinations, manque d'accès aux données propriétaires).
*   **Comparaison des approches d'adaptation :**
    *   *Fine-tuning* (ré-entraînement du modèle).
    *   *Prompt engineering* (optimisation des requêtes).
    *   *RAG* (intégration de contextes externes en temps réel).
*   Aperçu des architectures et variantes **RAG modernes** (Advanced RAG, re-ranking, etc.).

#### Étape 2.2 : Conception de l'Architecture et du Pipeline
Détailler la formalisation mathématique et technique du pipeline complet :
1.  **Encodage des Documents :** Transformation des morceaux de texte (chunks) en vecteurs (embeddings).
2.  **Indexation de la Base de Connaissances :** Stockage dans une base de données ou un index vectoriel.
3.  **Encodage de la Requête Utilisateur :** Vectorisation de la question posée dans le même espace latent.
4.  **Recherche Documentaire (Retrieval) :** Récupération des *top-k* documents ou extraits les plus pertinents via similarité cosinus ou produit scalaire.
5.  **Construction du Prompt Augmenté :** Concaténation de la requête initiale et du contenu récupéré.
6.  **Génération (LLM) :** Le modèle de langage génère une réponse documentée en se basant sur le contexte fourni.

#### Étape 2.3 : Implémentation Pratique
Développer un pipeline de RAG fonctionnel en utilisant les briques suivantes :
*   **Corpus de documents :** Extraire des données à partir de PDF (*comme fait avec `extract_pdf.py`*), d'articles ou de datasets préparés.
*   **Modèle d'Embeddings :** Utiliser des librairies comme `sentence-transformers` pour vectoriser le texte.
*   **Moteur de Recherche Vectoriel :** Utilisation de `FAISS` pour l'indexation rapide et le *similarity search*.
*   **Modèle Génératif :** Un modèle en local (ex. via HuggingFace) ou l'appel à une API externe.

#### Étape 2.4 : Démonstration et Expérimentation
*   Tester l'architecture métier avec différentes questions orientées sur le corpus spécifique configuré.
*   **Évaluation vs Baseline :** Comparer qualitativement (et quantitativement si possible) les réponses générées par le LLM **avec RAG** comparé au **LLM seul (sans RAG)** pour prouver son utilité.

---

## 📦 Livrables Attendus
1.  **Rapport PDF structuré et bien présenté :** Compilant l'état de l'art, les choix d'architecture, les résultats d'expérimentation et l'analyse.
2.  **Code Source :** Fourni sous forme de Notebooks (`.ipynb`) pour les démonstrations ou sous forme de projet Python bien modulaire (`/src/`, `/tests/`).
3.  **Démonstration Fonctionnelle :** Le système RAG doit pouvoir tourner et répondre avec succès lors d'une démonstration live.

---

*L'arborescence de test fournie (avec dossiers `src/rag`, `notebooks/`, `reports/`) est déjà parfaitement alignée avec ce cahier des charges !*
