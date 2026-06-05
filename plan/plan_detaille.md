# Plan détaillé — Mini-Projet NLP (ENSAH 2A ID)

---

## Phase 1 : Exploration des Transformers

### Objectif
Tester et évaluer 4 tâches NLP avec des vrais datasets et métriques standard.

### Étapes

#### 1. Setup
- `load_models.py` → télécharger les 8 modèles dans `models/`
- `load_datasets.py` → télécharger datasets + corpus RAG dans `data/`

#### 2. Notebook `01_exploration_transformers.ipynb`

| Tâche | Modèle | Dataset | Métrique |
|-------|--------|---------|----------|
| Classification zero-shot | `facebook/bart-large-mnli` | AG News (4 catégories) | Accuracy |
| Sentiment Analysis | `distilbert-base-uncased-finetuned-sst-2-english` | SST-2 | Accuracy, F1, Confusion Matrix |
| Question Answering | `deepset/roberta-base-squad2` | SQuAD v2 | Exact Match (EM), F1 |
| Résumé automatique | `facebook/bart-large-cnn` | CNN/DailyMail | ROUGE-1, ROUGE-2, ROUGE-L |

Chaque section du notebook contiendra :
- Description de la tâche
- Chargement du dataset
- Pipeline avec le modèle
- Évaluation quantitative
- Analyse des erreurs / exemples
- Visualisation (matplotlib)

#### 3. Bonus (si temps)
- Test du modèle multilingue `nlptown/bert-base-multilingual-uncased-sentiment` en français
- Comparaison des temps d'inférence CPU vs GPU

---

## Phase 2 : Pipeline RAG

### Objectif
Implémenter un système RAG complet et le comparer au LLM seul.

### Architecture

```
[PDF/Documents] → Chunking → Embeddings (all-MiniLM-L6-v2) → Index FAISS
                                                                    ↑
[Question utilisateur] → Embedding query → Similarity Search (top-k)
                                                      ↓
[Prompt augmenté] = Question + Contexte récupéré → LLM → Réponse
```

### Étapes

#### 1. Corpus
- Utiliser le corpus RAG de `load_datasets.py` (15 documents ML/NLP)
- OU extraire depuis des PDFs avec PyMuPDF
- Chunking : découpage en passages de ~256 tokens avec overlap

#### 2. Indexation
- Modèle d'embeddings : `sentence-transformers/all-MiniLM-L6-v2` (384 dims)
- Index vectoriel : FAISS `IndexFlatIP` (inner product = cosine sim)
- Sauvegarde de l'index dans `models/faiss_index/`

#### 3. Retrieval + Generation
- Encoder la question utilisateur
- Rechercher top-3 / top-5 documents
- Construire le prompt augmenté
- Générer réponse avec un LLM (local ou API)

#### 4. Comparaison Baseline vs RAG
- **Sans RAG** : Question directement au LLM → réponse (hallucinations possibles)
- **Avec RAG** : Question + contexte → réponse documentée
- Évaluation qualitative : les 2 réponses côte à côte
- Évaluation quantitative : faithfulness, pertinence (ou jugement manuel 1-5)

#### 5. Tests
- 5 à 10 questions sur le corpus
- Scénarios : question précise, question vague, question hors corpus

---

## Phase 3 : Livrables

### Rapport
- Notebook final nettoyé (markdown + résultats + graphes)
- Export PDF du notebook
- Sections : introduction, partie 1 (4 tâches), partie 2 (RAG), conclusion

### Code
- `src/exploration/` : scripts de chargement modèles/datasets
- `src/rag/` : pipeline RAG modulaire
- `notebooks/` : notebooks finalisés

### Démo
- Le système RAG doit répondre à des questions sur le corpus en live

---

## Calendrier suggéré

| Jour | Phase | Contenu |
|------|-------|---------|
| 1 | Setup | venv + scripts de chargement (déjà fait) |
| 2 | Phase 1 | Notebook exploration complet |
| 3 | Phase 2 | Pipeline RAG + indexation |
| 4 | Phase 2 | Comparaison baseline vs RAG + tests |
| 5 | Phase 3 | Finalisation notebook + rapport |
