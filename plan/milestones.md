# Plan de projet et jalons

## 1. Compréhension du projet

Le projet comporte deux parties principales :
- Exploration des Transformers et des tâches NLP : classification de texte, analyse de sentiment, question answering, résumé automatique.
- Implémentation d’un système RAG (Retrieval-Augmented Generation) avec un corpus de documents, embeddings, index vectoriel et modèle génératif.

## 2. Organisation des dossiers recommandée

- `plan/` : plan de projet, jalons, notes de conception.
- `data/` : corpus de documents, datasets, fichiers PDF ou articles.
- `src/` : code source Python réutilisable.
  - `src/exploration/` : scripts et notebooks d’exploration des Transformers.
  - `src/rag/` : implémentation du pipeline RAG (embeddings, index, récupération, génération).
- `notebooks/` : notebooks Jupyter pour démonstrations et essais.
- `models/` : modèles locaux, embeddings sauvegardés, index FAISS.
- `reports/` : rapport final, notes de travail, livrables PDF.
- `tests/` : tests d’intégration ou scripts de validation.

## 3. Jalons du projet

### Milestone 1a : Installation et configuration
Besoin :
- Environnement Python configuré.
- Librairies installées (`transformers`, `sentence-transformers`, `faiss`, `datasets`, `torch`, etc.).

### Milestone 1b : Exploration des Transformers et tâches NLP
Besoin :
- Charger des modèles pré-entraînés et tokenizers depuis le Model Hub.
- Tester des pipelines simples pour classification de texte, analyse de sentiment, question answering et résumé automatique.

### Milestone 2 : Recherche et conception du système RAG
Besoin :
- Comprendre la définition et les avantages des systèmes RAG.
- Identifier les limites des LLM sans données externes.
- Comparer les approches fine-tuning, prompt engineering et RAG.
- Formaliser le pipeline RAG : encodage documents, indexation, encodage requête, recherche top-k, construction du prompt, génération.

### Milestone 3 : Construction du pipeline RAG
Besoin :
- Collecter ou préparer un corpus de documents pertinents (extraction depuis des PDFs avec `extract_pdf.py` ou PyMuPDF).
- Générer des embeddings pour le corpus.
- Construire un index vectoriel (FAISS ou équivalent).
- Implémenter la recherche de documents pertinents.
- Intégrer un modèle génératif (API ou modèle local) pour répondre aux questions.
- Construire le prompt augmenté avec les documents récupérés.

### Milestone 4 : Tests, comparaison et validation
Besoin :
- Écrire des scénarios de test avec des questions sur le corpus.
- Comparer les réponses obtenues avec et sans RAG.
- Vérifier la qualité et la pertinence des résultats (évaluation qualitative + métriques quantitatives : BLEU, ROUGE, BERTScore, ou jugement manuel structuré 1-5).
- Mesurer la robustesse du système et corriger les problèmes.

### Milestone 5 : Documentation et livrables
Besoin :
- Rédiger un rapport PDF structuré.
- Documenter le code et le mode d’emploi.
- Préparer une démonstration fonctionnelle du système RAG.
- Organiser le projet pour livraison finale.
