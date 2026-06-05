"""
╔══════════════════════════════════════════════════════════════════╗
║         MINI-PROJET NLP — CHARGEMENT DES DATASETS               ║
║         ENSA Al Hoceima — Ingénierie des Données 2A              ║
╠══════════════════════════════════════════════════════════════════╣
║  Script : load_datasets.py                                       ║
║  Rôle   : Télécharger et sauvegarder tous les datasets           ║
║           dans le dossier data/                                  ║
║                                                                  ║
║  Datasets téléchargés :                                          ║
║  ┌─────────────────────────────────────────────────────────┐    ║
║  │ 1. sst2         → Sentiment (Stanford, 67K phrases)     │    ║
║  │ 2. ag_news      → Classification (4 catégories, 120K)   │    ║
║  │ 3. squad_v2     → Question Answering (130K paires)      │    ║
║  │ 4. cnn_dailymail → Résumé automatique (300K articles)   │    ║
║  │ 5. conll2003    → NER (4 types d'entités)               │    ║
║  │ 6. rag_corpus   → Corpus local pour RAG (custom)        │    ║
║  └─────────────────────────────────────────────────────────┘    ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import csv
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# ─── Configuration du logging ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ─── Imports ─────────────────────────────────────────────────────────────────
try:
    from datasets import load_dataset, DatasetDict, Dataset
    import pandas as pd
    import numpy as np
except ImportError as e:
    logger.error(f"Import échoué : {e}")
    logger.error("Installez : pip install datasets pandas numpy")
    sys.exit(1)


# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

BASE_DIR   = Path(__file__).resolve().parents[2]
DATA_DIR   = BASE_DIR / "data"

# Taille des échantillons pour ne pas surcharger le disque
SAMPLE_SIZES = {
    "train": 2000,
    "validation": 500,
    "test": 200,
}

# ─── Catalogue des datasets ──────────────────────────────────────────────────
DATASETS_CONFIG = {

    # ── 1. SST-2 : Sentiment Analysis ────────────────────────────────────────
    "sst2": {
        "hf_path":      "sst2",
        "hf_name":      None,
        "description":  "Stanford Sentiment Treebank v2. Phrases extraites de critiques "
                        "de cinéma annotées POSITIF/NÉGATIF. Dataset de référence pour "
                        "l'analyse de sentiment binaire.",
        "tache":        "sentiment",
        "splits":       ["train", "validation"],
        "colonnes":     {"texte": "sentence", "label": "label"},
        "classes":      ["NEGATIVE", "POSITIVE"],
        "taille":       "~870K tokens",
        "article":      "Socher et al. (2013) — Stanford NLP",
    },

    # ── 2. AG News : Classification de texte ─────────────────────────────────
    "ag_news": {
        "hf_path":      "ag_news",
        "hf_name":      None,
        "description":  "AG News Corpus. Articles de presse classés en 4 catégories : "
                        "World, Sports, Business, Science/Technology. 120K exemples "
                        "d'entraînement, 7.6K de test.",
        "tache":        "classification",
        "splits":       ["train", "test"],
        "colonnes":     {"texte": "text", "label": "label"},
        "classes":      ["World", "Sports", "Business", "Science/Tech"],
        "taille":       "~30MB",
        "article":      "Zhang et al. (2015)",
    },

    # ── 3. SQuAD v2 : Question Answering ─────────────────────────────────────
    "squad_v2": {
        "hf_path":      "squad_v2",
        "hf_name":      None,
        "description":  "Stanford Question Answering Dataset v2. 130K+ paires "
                        "question-contexte-réponse extraites de Wikipedia. SQuAD v2 "
                        "inclut des questions sans réponse (35% des cas).",
        "tache":        "question_answering",
        "splits":       ["train", "validation"],
        "colonnes":     {"texte": "context", "question": "question", "reponse": "answers"},
        "taille":       "~35MB",
        "article":      "Rajpurkar et al. (2016, 2018) — Stanford NLP",
    },

    # ── 4. CNN/DailyMail : Résumé automatique ────────────────────────────────
    "cnn_dailymail": {
        "hf_path":      "cnn_dailymail",
        "hf_name":      "3.0.0",
        "description":  "Dataset CNN/Daily Mail pour le résumé automatique. Articles "
                        "de presse avec leurs highlights comme résumés de référence. "
                        "300K+ paires article-résumé.",
        "tache":        "summarization",
        "splits":       ["train", "validation", "test"],
        "colonnes":     {"article": "article", "resume": "highlights"},
        "taille":       "~1.5GB",
        "article":      "Hermann et al. (2015) — DeepMind",
    },

    # ── 5. CoNLL-2003 : NER ───────────────────────────────────────────────────
    "conll2003": {
        "hf_path":      "conll2003",
        "hf_name":      None,
        "description":  "CoNLL-2003 Named Entity Recognition. Articles Reuters annotés "
                        "avec 4 types d'entités : PER (personnes), ORG (organisations), "
                        "LOC (lieux), MISC (divers). Schéma BIO.",
        "tache":        "ner",
        "splits":       ["train", "validation", "test"],
        "colonnes":     {"tokens": "tokens", "ner_tags": "ner_tags"},
        "classes":      ["O", "B-PER", "I-PER", "B-ORG", "I-ORG",
                         "B-LOC", "I-LOC", "B-MISC", "I-MISC"],
        "taille":       "~10MB",
        "article":      "Tjong Kim Sang & De Meulder (2003)",
    },
}


# ════════════════════════════════════════════════════════════════════════════
# CORPUS RAG LOCAL (créé sans téléchargement)
# ════════════════════════════════════════════════════════════════════════════

RAG_CORPUS = [
    {
        "id": 1, "categorie": "machine_learning",
        "titre": "Introduction au Machine Learning",
        "texte": (
            "Machine learning is a subset of artificial intelligence that enables systems "
            "to learn and improve from experience without being explicitly programmed. "
            "It focuses on developing computer programs that can access data and use it "
            "to learn for themselves. The process begins with observations or data — "
            "examples, direct experience, or instruction — to look for patterns in data "
            "and make better decisions in the future. The primary goal is to allow computers "
            "to learn automatically without human intervention or assistance and adjust "
            "actions accordingly."
        ),
    },
    {
        "id": 2, "categorie": "machine_learning",
        "titre": "Types d'apprentissage automatique",
        "texte": (
            "There are three main types of machine learning: supervised learning, "
            "unsupervised learning, and reinforcement learning. Supervised learning uses "
            "labeled training data to train models — examples include linear regression, "
            "decision trees, and neural networks. Unsupervised learning finds hidden "
            "patterns in unlabeled data using clustering (K-Means, DBSCAN) and "
            "dimensionality reduction (PCA, t-SNE). Reinforcement learning involves an "
            "agent interacting with an environment, learning to maximize cumulative rewards "
            "through trial and error — used in robotics and game playing."
        ),
    },
    {
        "id": 3, "categorie": "deep_learning",
        "titre": "Réseaux de Neurones Profonds",
        "texte": (
            "Deep learning is a machine learning technique that uses artificial neural "
            "networks with many layers (deep architectures) to learn hierarchical "
            "representations. Each layer transforms the input into progressively more "
            "abstract representations. Deep learning powers image recognition, speech "
            "recognition, natural language processing, and autonomous driving. The key "
            "components are: neurons (units computing weighted sums), activation functions "
            "(ReLU, sigmoid, tanh), backpropagation for gradient computation, and "
            "stochastic gradient descent for optimization."
        ),
    },
    {
        "id": 4, "categorie": "deep_learning",
        "titre": "CNN et RNN — Architectures spécialisées",
        "texte": (
            "Convolutional Neural Networks (CNNs) are designed for grid-structured data "
            "like images. They use convolutional filters to detect local patterns, pooling "
            "layers to reduce dimensionality, and fully-connected layers for classification. "
            "Famous CNNs include LeNet, AlexNet, VGG, ResNet, and EfficientNet. "
            "Recurrent Neural Networks (RNNs) process sequential data by maintaining a "
            "hidden state. LSTM (Long Short-Term Memory) networks use gates (input, forget, "
            "output) to control information flow, solving the vanishing gradient problem "
            "that affects vanilla RNNs."
        ),
    },
    {
        "id": 5, "categorie": "transformers",
        "titre": "Architecture Transformer",
        "texte": (
            "The Transformer architecture was introduced by Vaswani et al. in the paper "
            "'Attention is All You Need' (2017). It replaces recurrence with self-attention "
            "mechanisms, allowing parallel processing of entire sequences. The encoder "
            "consists of N identical layers, each with multi-head self-attention and a "
            "feed-forward network. The decoder adds a cross-attention layer to attend to "
            "encoder output. Positional encodings inject position information since the "
            "model has no inherent notion of order. Multi-head attention computes attention "
            "in h parallel 'heads', each learning different relationships."
        ),
    },
    {
        "id": 6, "categorie": "transformers",
        "titre": "BERT — Représentations bidirectionnelles",
        "texte": (
            "BERT (Bidirectional Encoder Representations from Transformers) was introduced "
            "by Devlin et al. at Google AI in 2018. Unlike GPT which reads text left-to-right, "
            "BERT reads text in both directions simultaneously, enabling richer context "
            "understanding. BERT is pre-trained on two tasks: Masked Language Modeling (MLM) "
            "— predicting randomly masked tokens — and Next Sentence Prediction (NSP). "
            "After pre-training on 3.3B words, BERT is fine-tuned for downstream tasks. "
            "BERT-Base has 12 layers, 768 hidden dimensions, 12 attention heads (110M params). "
            "BERT-Large has 24 layers, 1024 hidden dimensions, 16 heads (340M params)."
        ),
    },
    {
        "id": 7, "categorie": "rag",
        "titre": "RAG — Retrieval-Augmented Generation",
        "texte": (
            "Retrieval-Augmented Generation (RAG) is an AI framework combining information "
            "retrieval with text generation, proposed by Lewis et al. at Facebook AI "
            "Research in 2020. RAG addresses LLM limitations: knowledge cutoff (frozen "
            "training knowledge), hallucinations (generating false facts confidently), "
            "and lack of source attribution. The RAG pipeline: (1) index documents as "
            "dense vector embeddings, (2) encode user query, (3) retrieve top-k similar "
            "documents via vector similarity search, (4) build augmented prompt with "
            "retrieved context, (5) generate answer with LLM. This significantly reduces "
            "hallucinations and enables up-to-date knowledge access."
        ),
    },
    {
        "id": 8, "categorie": "rag",
        "titre": "Embeddings et Recherche Sémantique",
        "texte": (
            "Word embeddings represent words as dense vectors in a continuous vector space "
            "where semantically similar words are geometrically close. Word2Vec (2013) uses "
            "CBOW and Skip-gram architectures. GloVe uses global co-occurrence statistics. "
            "Sentence Transformers (SBERT, 2019) fine-tune BERT with siamese networks to "
            "produce meaningful sentence embeddings. Cosine similarity measures the angle "
            "between vectors: sim(A,B) = (A·B)/(|A||B|). Values close to 1 indicate high "
            "semantic similarity. For RAG, embeddings of size 384-768 dimensions are typical, "
            "capturing rich semantic representations for retrieval."
        ),
    },
    {
        "id": 9, "categorie": "rag",
        "titre": "FAISS — Recherche vectorielle efficace",
        "texte": (
            "FAISS (Facebook AI Similarity Search) is a library for efficient similarity "
            "search and clustering of dense vectors, developed by Facebook AI Research. "
            "It supports multiple index types: IndexFlatL2 and IndexFlatIP for exact search "
            "(brute force), IndexIVFFlat for approximate search using inverted file indexing "
            "(clusters vectors into Voronoi cells), IndexHNSWFlat using hierarchical "
            "navigable small world graphs for extremely fast approximate search. For billion-"
            "scale datasets, product quantization (PQ) compresses vectors. FAISS can perform "
            "exact nearest-neighbor search on 1M vectors in under 1 second on CPU."
        ),
    },
    {
        "id": 10, "categorie": "nlp_general",
        "titre": "Large Language Models (LLMs)",
        "texte": (
            "Large Language Models (LLMs) are transformer-based models trained on massive "
            "text corpora to predict the next token. GPT-3 (175B parameters) demonstrated "
            "remarkable few-shot learning: performing tasks with just a few examples in the "
            "prompt, without gradient updates. GPT-4 further improved reasoning. Open-source "
            "alternatives include LLaMA (Meta), Mistral, and Falcon. Key concepts: "
            "in-context learning (learning from prompt examples), chain-of-thought prompting "
            "(step-by-step reasoning), and instruction tuning (RLHF — Reinforcement Learning "
            "from Human Feedback) to align model behavior with human preferences."
        ),
    },
    {
        "id": 11, "categorie": "nlp_general",
        "titre": "Métriques d'évaluation NLP",
        "texte": (
            "NLP evaluation metrics measure model quality on specific tasks. BLEU (Bilingual "
            "Evaluation Understudy) measures n-gram precision between generated and reference "
            "text — standard for machine translation. ROUGE (Recall-Oriented Understudy for "
            "Gisting Evaluation) measures recall-oriented n-gram overlap, used for "
            "summarization (ROUGE-1, ROUGE-2, ROUGE-L). Perplexity measures how well a "
            "language model predicts a text sample. For RAG systems: faithfulness (responses "
            "grounded in retrieved context), answer relevancy (response addresses the query), "
            "context recall (relevant information present in retrieved documents), and "
            "hallucination rate (factual errors)."
        ),
    },
    {
        "id": 12, "categorie": "nlp_general",
        "titre": "Applications pratiques du NLP",
        "texte": (
            "NLP powers numerous real-world applications across industries. Machine translation "
            "(Google Translate, DeepL) achieves near-human quality for major language pairs. "
            "Virtual assistants (Siri, Alexa, Google Assistant) combine ASR, NLU, and NLG. "
            "In healthcare, NLP extracts structured information from clinical notes (ICD "
            "coding, adverse event detection). In finance, sentiment analysis of news and "
            "social media drives algorithmic trading. Legal NLP automates contract review "
            "and case research. Code generation models (GitHub Copilot, Code Llama) assist "
            "developers. Customer service chatbots handle 80% of routine inquiries automatically."
        ),
    },
    {
        "id": 13, "categorie": "transformers",
        "titre": "Variantes de BERT et modèles dérivés",
        "texte": (
            "After BERT, numerous variants improved specific aspects. RoBERTa (2019, Facebook) "
            "uses longer training, larger batches, dynamic masking, and removes NSP — achieving "
            "better performance. DistilBERT (2019, HuggingFace) distills BERT into a 40% "
            "smaller, 60% faster model retaining 97% of performance. ALBERT uses parameter "
            "sharing and factorized embeddings for efficiency. XLNet uses permutation language "
            "modeling. DeBERTa uses disentangled attention with separate position encodings. "
            "For generation: GPT-2 and GPT-3 (OpenAI), BART combines BERT-like encoder with "
            "GPT-like decoder, and T5 (Google) frames all NLP tasks as text-to-text."
        ),
    },
    {
        "id": 14, "categorie": "rag",
        "titre": "Variantes modernes de RAG",
        "texte": (
            "Modern RAG architectures go beyond simple retrieve-then-generate pipelines. "
            "Modular RAG introduces flexible components: query rewriting (improves retrieval), "
            "reranking (cross-encoder reorders results), and fusion (merges multiple retrievals). "
            "Corrective RAG (CRAG) evaluates retrieved document quality and triggers web search "
            "when documents are insufficient. Self-RAG trains the model to decide when to "
            "retrieve and how to use results. GraphRAG (Microsoft, 2024) uses knowledge graphs "
            "for complex multi-hop reasoning. Agentic RAG uses autonomous agents that plan "
            "multi-step retrieval strategies. HyDE generates hypothetical documents to improve "
            "semantic matching for sparse queries."
        ),
    },
    {
        "id": 15, "categorie": "deep_learning",
        "titre": "Optimisation et régularisation",
        "texte": (
            "Training deep neural networks requires careful optimization and regularization. "
            "Optimizers: SGD with momentum, Adam (adaptive learning rates per parameter), "
            "AdamW (Adam with weight decay decoupling), and LAMB for large-batch training. "
            "Learning rate scheduling: warmup (gradual increase from 0), cosine annealing, "
            "and ReduceLROnPlateau. Regularization techniques: dropout (randomly zeroing "
            "activations during training), L1/L2 weight regularization, batch normalization "
            "(normalizes layer inputs), layer normalization (used in Transformers), and "
            "gradient clipping (prevents exploding gradients). Data augmentation and early "
            "stopping prevent overfitting on limited datasets."
        ),
    },
]


# ════════════════════════════════════════════════════════════════════════════
# UTILITAIRES
# ════════════════════════════════════════════════════════════════════════════

def creer_arborescence():
    """Crée l'arborescence du dossier data/."""
    dossiers = [
        DATA_DIR,
        DATA_DIR / "raw",
        DATA_DIR / "processed",
        DATA_DIR / "rag_corpus",
        DATA_DIR / "splits",
        DATA_DIR / "stats",
    ]
    for d in dossiers:
        d.mkdir(parents=True, exist_ok=True)
    logger.info(f"✅ Arborescence créée dans : {DATA_DIR}")


def sauvegarder_json(data, path: Path, nom: str = ""):
    """Sauvegarde des données en JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"💾 Sauvegardé ({nom}) → {path}")


def sauvegarder_csv(records: list, path: Path, nom: str = ""):
    """Sauvegarde une liste de dicts en CSV."""
    if not records:
        return
    df = pd.DataFrame(records)
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info(f"💾 CSV sauvegardé ({nom}, {len(records)} lignes) → {path}")


def afficher_stats_dataset(df: pd.DataFrame, nom: str, colonne_label: str = None):
    """Affiche des statistiques descriptives sur le dataset."""
    print(f"\n  📊 Stats — {nom}")
    print(f"     Nb exemples : {len(df):,}")

    if colonne_label and colonne_label in df.columns:
        print(f"     Distribution des classes :")
        vc = df[colonne_label].value_counts()
        for cls, cnt in vc.items():
            pct = cnt / len(df) * 100
            print(f"       {str(cls):<15} : {cnt:>6,} ({pct:.1f}%)")

    # Longueur des textes
    for col in df.columns:
        if df[col].dtype == object:
            longueurs = df[col].dropna().str.split().str.len()
            if len(longueurs) > 0:
                print(f"     Longueur '{col}' : moy={longueurs.mean():.0f} | "
                      f"min={longueurs.min()} | max={longueurs.max()}")


# ════════════════════════════════════════════════════════════════════════════
# FONCTIONS DE TÉLÉCHARGEMENT PAR TÂCHE
# ════════════════════════════════════════════════════════════════════════════

def telecharger_sst2(config: dict) -> dict:
    """Télécharge et prépare le dataset SST-2."""
    logger.info("📥 Téléchargement SST-2...")
    raw = load_dataset(config["hf_path"])
    save_dir = DATA_DIR / "raw" / "sst2"
    save_dir.mkdir(exist_ok=True)

    records = {"train": [], "validation": []}
    for split in ["train", "validation"]:
        data = raw[split]
        n = min(SAMPLE_SIZES[split], len(data))
        sample = data.select(range(n))
        records[split] = [
            {
                "texte": ex["sentence"],
                "label": ex["label"],
                "label_nom": config["classes"][ex["label"]],
            }
            for ex in sample
        ]
        sauvegarder_json(records[split], save_dir / f"{split}.json", f"sst2/{split}")
        sauvegarder_csv(records[split], save_dir / f"{split}.csv", f"sst2/{split}")

    # Stats globales
    all_records = records["train"] + records["validation"]
    df = pd.DataFrame(all_records)
    afficher_stats_dataset(df, "SST-2", "label_nom")
    return {"dataset": "sst2", "train": len(records["train"]),
            "validation": len(records["validation"]), "status": "ok"}


def telecharger_ag_news(config: dict) -> dict:
    """Télécharge et prépare AG News."""
    logger.info("📥 Téléchargement AG News...")
    raw = load_dataset(config["hf_path"])
    save_dir = DATA_DIR / "raw" / "ag_news"
    save_dir.mkdir(exist_ok=True)

    records = {}
    for split in ["train", "test"]:
        data = raw[split]
        key = "test" if split == "test" else "train"
        n = min(SAMPLE_SIZES[key], len(data))
        sample = data.select(range(n))
        records[split] = [
            {
                "texte": ex["text"],
                "label": ex["label"],
                "label_nom": config["classes"][ex["label"]],
            }
            for ex in sample
        ]
        sauvegarder_json(records[split], save_dir / f"{split}.json", f"ag_news/{split}")
        sauvegarder_csv(records[split], save_dir / f"{split}.csv", f"ag_news/{split}")

    df = pd.DataFrame(records["train"] + records["test"])
    afficher_stats_dataset(df, "AG News", "label_nom")
    return {"dataset": "ag_news", **{k: len(v) for k, v in records.items()}, "status": "ok"}


def telecharger_squad(config: dict) -> dict:
    """Télécharge et prépare SQuAD v2."""
    logger.info("📥 Téléchargement SQuAD v2...")
    raw = load_dataset(config["hf_path"])
    save_dir = DATA_DIR / "raw" / "squad_v2"
    save_dir.mkdir(exist_ok=True)

    records = {}
    for split in ["train", "validation"]:
        data = raw[split]
        n = min(SAMPLE_SIZES[split], len(data))
        sample = data.select(range(n))
        records[split] = []
        for ex in sample:
            # Extraire la première réponse (ou vide si sans réponse)
            answers = ex["answers"]
            reponse_texte = answers["text"][0] if answers["text"] else ""
            a_reponse = len(answers["text"]) > 0
            records[split].append({
                "id":       ex["id"],
                "titre":    ex["title"],
                "contexte": ex["context"],
                "question": ex["question"],
                "reponse":  reponse_texte,
                "a_reponse": a_reponse,
            })
        sauvegarder_json(records[split], save_dir / f"{split}.json", f"squad/{split}")

    df = pd.DataFrame(records["train"])
    afficher_stats_dataset(df, "SQuAD v2", "a_reponse")
    return {"dataset": "squad_v2", **{k: len(v) for k, v in records.items()}, "status": "ok"}


def telecharger_cnn_dailymail(config: dict) -> dict:
    """Télécharge et prépare CNN/DailyMail."""
    logger.info("📥 Téléchargement CNN/DailyMail...")
    raw = load_dataset(config["hf_path"], config["hf_name"])
    save_dir = DATA_DIR / "raw" / "cnn_dailymail"
    save_dir.mkdir(exist_ok=True)

    records = {}
    for split in ["train", "validation", "test"]:
        data = raw[split]
        key = split if split in SAMPLE_SIZES else "test"
        n = min(SAMPLE_SIZES.get(key, 100), len(data))
        sample = data.select(range(n))
        records[split] = [
            {
                "id":      ex.get("id", f"{split}_{i}"),
                "article": ex["article"],
                "resume":  ex["highlights"],
                "nb_mots_article": len(ex["article"].split()),
                "nb_mots_resume":  len(ex["highlights"].split()),
            }
            for i, ex in enumerate(sample)
        ]
        sauvegarder_json(records[split], save_dir / f"{split}.json", f"cnn/{split}")

    df = pd.DataFrame(records["train"])
    afficher_stats_dataset(df, "CNN/DailyMail")
    return {"dataset": "cnn_dailymail", **{k: len(v) for k, v in records.items()}, "status": "ok"}


def telecharger_conll2003(config: dict) -> dict:
    """Télécharge et prépare CoNLL-2003."""
    logger.info("📥 Téléchargement CoNLL-2003...")
    raw = load_dataset(config["hf_path"], trust_remote_code=True)
    save_dir = DATA_DIR / "raw" / "conll2003"
    save_dir.mkdir(exist_ok=True)

    # Mapping des IDs vers les noms de classes
    tag_names = config["classes"]

    records = {}
    for split in ["train", "validation", "test"]:
        data = raw[split]
        n = min(SAMPLE_SIZES.get(split, 200), len(data))
        sample = data.select(range(n))
        records[split] = [
            {
                "tokens":    ex["tokens"],
                "ner_tags":  ex["ner_tags"],
                "ner_names": [tag_names[t] for t in ex["ner_tags"]],
                "nb_tokens": len(ex["tokens"]),
            }
            for ex in sample
        ]
        sauvegarder_json(records[split], save_dir / f"{split}.json", f"conll/{split}")

    print(f"\n  📊 Stats — CoNLL-2003")
    print(f"     Nb phrases : train={len(records['train'])}, val={len(records['validation'])}")
    all_tags = [t for r in records["train"] for t in r["ner_names"]]
    from collections import Counter
    tag_counts = Counter(all_tags)
    for tag, cnt in tag_counts.most_common(5):
        print(f"       {tag:<12} : {cnt:>5,} tokens")
    return {"dataset": "conll2003", **{k: len(v) for k, v in records.items()}, "status": "ok"}


def creer_corpus_rag() -> dict:
    """Crée et sauvegarde le corpus local pour le système RAG."""
    logger.info("✍️  Création du corpus RAG local...")
    save_dir = DATA_DIR / "rag_corpus"

    # Sauvegarde principale
    sauvegarder_json(RAG_CORPUS, save_dir / "corpus.json", "rag_corpus")
    sauvegarder_csv(RAG_CORPUS, save_dir / "corpus.csv", "rag_corpus")

    # Index par catégorie
    categories = {}
    for doc in RAG_CORPUS:
        cat = doc["categorie"]
        categories.setdefault(cat, []).append(doc)

    sauvegarder_json(categories, save_dir / "corpus_par_categorie.json", "rag_corpus_categories")

    # Metadata
    meta = {
        "nb_documents": len(RAG_CORPUS),
        "categories":   {cat: len(docs) for cat, docs in categories.items()},
        "total_mots":   sum(len(d["texte"].split()) for d in RAG_CORPUS),
        "date_creation": datetime.now().isoformat(),
    }
    sauvegarder_json(meta, save_dir / "metadata.json", "rag_meta")

    print(f"\n  📊 Stats — Corpus RAG")
    print(f"     Nb documents  : {meta['nb_documents']}")
    print(f"     Total mots    : {meta['total_mots']:,}")
    print(f"     Catégories    :")
    for cat, n in meta["categories"].items():
        print(f"       {cat:<25} : {n} documents")

    return {"dataset": "rag_corpus", "nb_docs": len(RAG_CORPUS), "status": "ok"}


# ════════════════════════════════════════════════════════════════════════════
# STATISTIQUES GLOBALES
# ════════════════════════════════════════════════════════════════════════════

def generer_rapport_stats(resultats: list):
    """Génère un rapport global sur tous les datasets téléchargés."""
    rapport = {
        "date_generation": datetime.now().isoformat(),
        "datasets": resultats,
        "resume": {
            "total_datasets": len(resultats),
            "succes": sum(1 for r in resultats if r.get("status") == "ok"),
            "erreurs": sum(1 for r in resultats if r.get("status") == "erreur"),
        }
    }
    path = DATA_DIR / "stats" / "rapport_datasets.json"
    sauvegarder_json(rapport, path, "rapport")

    # README automatique
    readme = ["# 📊 Datasets — Mini-Projet NLP\n\n"]
    readme.append(f"Générés le : {rapport['date_generation']}\n\n")
    readme.append("## Datasets disponibles\n\n")
    readme.append("| Dataset | Tâche | Train | Validation | Test |\n")
    readme.append("|---------|-------|-------|-----------|------|\n")
    for r in resultats:
        train = r.get("train", "—")
        val   = r.get("validation", "—")
        test  = r.get("test", "—")
        tache = DATASETS_CONFIG.get(r["dataset"], {}).get("tache", r["dataset"])
        readme.append(f"| {r['dataset']} | {tache} | {train} | {val} | {test} |\n")
    readme.append("\n## Structure\n```\ndata/\n├── raw/          # Données brutes par dataset\n")
    readme.append("├── processed/    # Données prétraitées\n├── rag_corpus/   # Corpus RAG\n└── stats/        # Statistiques\n```\n")

    with open(DATA_DIR / "README.md", "w", encoding="utf-8") as f:
        f.writelines(readme)

    logger.info(f"📄 Rapport généré → {path}")


# ════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ════════════════════════════════════════════════════════════════════════════

# Dispatch des téléchargeurs
TELECHARGEURS = {
    "sst2":         telecharger_sst2,
    "ag_news":      telecharger_ag_news,
    "squad_v2":     telecharger_squad,
    "cnn_dailymail": telecharger_cnn_dailymail,
    "conll2003":    telecharger_conll2003,
}


def main(datasets_specifiques: list = None, inclure_rag: bool = True):
    """
    Télécharge tous les datasets configurés dans data/.

    Paramètres
    ----------
    datasets_specifiques : list ou None
        Si fourni, ne télécharge que ces datasets.
    inclure_rag : bool
        Inclure la création du corpus RAG local.
    """
    print("╔" + "═" * 58 + "╗")
    print("║  MINI-PROJET NLP — CHARGEMENT DES DATASETS                ║")
    print("║  ENSA Al Hoceima — Ingénierie des Données 2A              ║")
    print("╚" + "═" * 58 + "╝")
    print(f"\n  Dossier       : {DATA_DIR}")
    print(f"  Tailles       : train={SAMPLE_SIZES['train']}, "
          f"val={SAMPLE_SIZES['validation']}, test={SAMPLE_SIZES['test']}")

    creer_arborescence()

    # Sélection des datasets
    datasets_a_charger = {
        k: v for k, v in DATASETS_CONFIG.items()
        if datasets_specifiques is None or k in datasets_specifiques
    }

    print(f"\n  Datasets à télécharger ({len(datasets_a_charger)}) :")
    for i, (nom, cfg) in enumerate(datasets_a_charger.items(), 1):
        print(f"  {i}. {nom:<20} — {cfg['tache']:<25} ({cfg['taille']})")

    t_global = time.time()
    resultats = []

    for nom, config in datasets_a_charger.items():
        print(f"\n{'═'*60}")
        print(f"  📦 {nom.upper()}")
        print("═" * 60)
        print(f"  Description : {config['description'][:100]}...")

        t0 = time.time()
        try:
            chargeur = TELECHARGEURS[nom]
            resultat = chargeur(config)
            duree = time.time() - t0
            resultat["duree_s"] = round(duree, 2)
            print(f"\n  ✅ Terminé en {duree:.1f}s")
        except Exception as e:
            duree = time.time() - t0
            logger.error(f"❌ Erreur pour {nom} : {e}")
            resultat = {"dataset": nom, "status": "erreur",
                        "erreur": str(e), "duree_s": round(duree, 2)}
        resultats.append(resultat)

    # Corpus RAG
    if inclure_rag:
        print(f"\n{'═'*60}")
        print(f"  📦 CORPUS RAG (LOCAL)")
        print("═" * 60)
        try:
            res = creer_corpus_rag()
            resultats.append(res)
        except Exception as e:
            logger.error(f"❌ Corpus RAG : {e}")

    # Résumé final
    duree_totale = time.time() - t_global
    n_ok     = sum(1 for r in resultats if r.get("status") == "ok")
    n_erreur = sum(1 for r in resultats if r.get("status") == "erreur")

    print(f"\n{'═'*60}")
    print("  RÉSUMÉ FINAL")
    print("═" * 60)
    print(f"  ✅ Succès  : {n_ok}")
    print(f"  ❌ Erreurs : {n_erreur}")
    print(f"  ⏱️  Durée  : {duree_totale:.1f}s")

    for r in resultats:
        icon = "✅" if r.get("status") == "ok" else "❌"
        print(f"  {icon} {r['dataset']:<20} → {r.get('duree_s', '?')}s")

    generer_rapport_stats(resultats)
    print(f"\n✅ Données disponibles dans : {DATA_DIR}")
    return resultats


# ════════════════════════════════════════════════════════════════════════════
# FONCTIONS HELPER — Chargement depuis disque
# ════════════════════════════════════════════════════════════════════════════

def charger_dataset_local(nom: str, split: str = "train") -> pd.DataFrame:
    """
    Charge un dataset déjà téléchargé depuis data/raw/.

    Usage:
    >>> from src.exploration.load_datasets import charger_dataset_local
    >>> df = charger_dataset_local("sst2", "train")
    """
    mapping = {
        "sst2":         DATA_DIR / "raw" / "sst2"         / f"{split}.json",
        "ag_news":      DATA_DIR / "raw" / "ag_news"      / f"{split}.json",
        "squad_v2":     DATA_DIR / "raw" / "squad_v2"     / f"{split}.json",
        "cnn_dailymail": DATA_DIR / "raw" / "cnn_dailymail"/ f"{split}.json",
        "conll2003":    DATA_DIR / "raw" / "conll2003"    / f"{split}.json",
    }
    path = mapping.get(nom)
    if path is None:
        raise ValueError(f"Dataset inconnu : {nom}")
    if not path.exists():
        raise FileNotFoundError(f"Non trouvé : {path}\nLancez d'abord : python load_datasets.py")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)


def charger_corpus_rag() -> list:
    """Charge le corpus RAG depuis data/rag_corpus/corpus.json."""
    path = DATA_DIR / "rag_corpus" / "corpus.json"
    if not path.exists():
        raise FileNotFoundError(f"Corpus RAG non trouvé : {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Télécharge les datasets NLP dans data/"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help=f"Datasets à télécharger. Options : {list(DATASETS_CONFIG.keys())}"
    )
    parser.add_argument(
        "--no-rag", action="store_true",
        help="Ne pas créer le corpus RAG local"
    )
    parser.add_argument(
        "--sizes", nargs=3, type=int, default=None,
        metavar=("TRAIN", "VAL", "TEST"),
        help="Tailles des splits (défaut: 2000 500 200)"
    )
    args = parser.parse_args()

    if args.sizes:
        SAMPLE_SIZES["train"]      = args.sizes[0]
        SAMPLE_SIZES["validation"] = args.sizes[1]
        SAMPLE_SIZES["test"]       = args.sizes[2]

    main(
        datasets_specifiques=args.datasets,
        inclure_rag=not args.no_rag,
    )