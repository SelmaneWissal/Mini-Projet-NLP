"""
╔══════════════════════════════════════════════════════════════════╗
║         MINI-PROJET NLP — CHARGEMENT DES MODÈLES                ║
║         ENSA Al Hoceima — Ingénierie des Données 2A              ║
╠══════════════════════════════════════════════════════════════════╣
║  Script : load_models.py                                         ║
║  Rôle   : Télécharger et sauvegarder tous les modèles            ║
║           HuggingFace dans le dossier models/                    ║
║                                                                  ║
║  Modèles téléchargés :                                           ║
║  ┌─────────────────────────────────────────────────────────┐    ║
║  │ 1. bert-base-uncased          → Tokenizer & Encodeur    │    ║
║  │ 2. distilbert-sst2            → Analyse de sentiment    │    ║
║  │ 3. nlptown/bert-multilingual  → Sentiment multilingue   │    ║
║  │ 4. facebook/bart-large-mnli   → Classification ZS       │    ║
║  │ 5. deepset/roberta-squad2     → Question Answering      │    ║
║  │ 6. facebook/bart-large-cnn    → Résumé automatique      │    ║
║  │ 7. dbmdz/bert-large-conll03   → NER                     │    ║
║  │ 8. all-MiniLM-L6-v2          → Embeddings RAG           │    ║
║  └─────────────────────────────────────────────────────────┘    ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime

# ─── Configuration du logging ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("models/download_log.txt", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# ─── Imports HuggingFace ─────────────────────────────────────────────────────
try:
    from transformers import (
        AutoTokenizer,
        AutoModel,
        AutoModelForSequenceClassification,
        AutoModelForQuestionAnswering,
        AutoModelForSeq2SeqLM,
        AutoModelForTokenClassification,
        pipeline,
    )
    from sentence_transformers import SentenceTransformer
    import torch
except ImportError as e:
    logger.error(f"Import échoué : {e}")
    logger.error("Installez les dépendances : pip install transformers torch sentence-transformers")
    sys.exit(1)


# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION DES MODÈLES
# ════════════════════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).resolve().parents[2]   # racine du projet
MODELS_DIR = BASE_DIR / "models"

# Dictionnaire de tous les modèles à télécharger
# Format : { "nom_local" : { "model_id": ..., "type": ..., "tache": ..., "description": ... } }
MODELS_CONFIG = {

    # ── 1. BERT BASE (tokenizer de référence) ────────────────────────────────
    "bert-base-uncased": {
        "model_id": "bert-base-uncased",
        "type": "auto",                        # AutoTokenizer + AutoModel
        "tache": "encodage",
        "description": "Modèle BERT de base non casé — utilisé comme tokenizer de "
                        "référence et pour les embeddings contextuels. Entraîné sur "
                        "BooksCorpus + Wikipedia (3.3B tokens). 110M paramètres.",
        "taille_approx": "440 MB",
    },

    # ── 2. DistilBERT Sentiment (SST-2) ──────────────────────────────────────
    "distilbert-sentiment": {
        "model_id": "distilbert-base-uncased-finetuned-sst-2-english",
        "type": "sequence_classification",
        "tache": "sentiment",
        "description": "DistilBERT fine-tuné sur SST-2 (Stanford Sentiment Treebank). "
                        "60% plus rapide que BERT, 40% moins de paramètres, conserve "
                        "97% des performances. Prédit POSITIVE / NEGATIVE.",
        "taille_approx": "255 MB",
    },

    # ── 3. BERT Multilingual Sentiment ───────────────────────────────────────
    "bert-multilingual-sentiment": {
        "model_id": "nlptown/bert-base-multilingual-uncased-sentiment",
        "type": "sequence_classification",
        "tache": "sentiment_multilingue",
        "description": "BERT multilingue fine-tuné pour l'analyse de sentiment. "
                        "Supporte 6 langues : EN, FR, DE, ES, IT, NL. "
                        "Prédit de 1 à 5 étoiles. Idéal pour les avis produits.",
        "taille_approx": "680 MB",
    },

    # ── 4. BART MNLI (Zero-Shot Classification) ───────────────────────────────
    "bart-large-mnli": {
        "model_id": "facebook/bart-large-mnli",
        "type": "sequence_classification",
        "tache": "classification_zero_shot",
        "description": "BART Large fine-tuné sur MNLI pour la classification zéro-shot. "
                        "Utilise l'inférence en langage naturel (NLI) pour classifier "
                        "sans données d'entraînement spécifiques. 406M paramètres.",
        "taille_approx": "1.6 GB",
    },

    # ── 5. RoBERTa SQuAD2 (Question Answering) ────────────────────────────────
    "roberta-squad2": {
        "model_id": "deepset/roberta-base-squad2",
        "type": "question_answering",
        "tache": "question_answering",
        "description": "RoBERTa Base fine-tuné sur SQuAD v2. Supporte les questions "
                        "sans réponse dans le contexte. Prédit les positions de début "
                        "et de fin de la réponse dans le texte.",
        "taille_approx": "480 MB",
    },

    # ── 6. BART CNN (Résumé Automatique) ──────────────────────────────────────
    "bart-large-cnn": {
        "model_id": "facebook/bart-large-cnn",
        "type": "seq2seq",
        "tache": "resume",
        "description": "BART Large fine-tuné sur CNN/DailyMail pour le résumé abstractif. "
                        "Génère des résumés fluides et cohérents. Utilise beam search "
                        "pour la génération. 406M paramètres.",
        "taille_approx": "1.6 GB",
    },

    # ── 7. BERT CoNLL03 (NER) ─────────────────────────────────────────────────
    "bert-ner-conll03": {
        "model_id": "dbmdz/bert-large-cased-finetuned-conll03-english",
        "type": "token_classification",
        "tache": "ner",
        "description": "BERT Large Cased fine-tuné sur CoNLL-2003 pour la reconnaissance "
                        "d'entités nommées. Détecte PER (personnes), ORG (organisations), "
                        "LOC (lieux), MISC (divers). F1-score de 92.4%.",
        "taille_approx": "1.3 GB",
    },

    # ── 8. Sentence Transformer (Embeddings RAG) ──────────────────────────────
    "all-MiniLM-L6-v2": {
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "type": "sentence_transformer",
        "tache": "embeddings_rag",
        "description": "Sentence Transformer léger et rapide. Produit des embeddings "
                        "de 384 dimensions pour phrases et paragraphes. Excellent pour "
                        "la recherche sémantique et le système RAG. 22M paramètres.",
        "taille_approx": "90 MB",
    },
}


# ════════════════════════════════════════════════════════════════════════════
# FONCTIONS UTILITAIRES
# ════════════════════════════════════════════════════════════════════════════

def creer_arborescence():
    """Crée les dossiers nécessaires dans models/."""
    dossiers = [
        MODELS_DIR,
        MODELS_DIR / "tokenizers",
        MODELS_DIR / "classification",
        MODELS_DIR / "sentiment",
        MODELS_DIR / "question_answering",
        MODELS_DIR / "summarization",
        MODELS_DIR / "ner",
        MODELS_DIR / "embeddings",
        MODELS_DIR / "logs",
    ]
    for d in dossiers:
        d.mkdir(parents=True, exist_ok=True)
    logger.info(f"✅ Arborescence créée dans : {MODELS_DIR}")


def get_save_path(nom_local: str, tache: str) -> Path:
    """Retourne le chemin de sauvegarde selon la tâche."""
    mapping = {
        "encodage":               MODELS_DIR / "tokenizers"       / nom_local,
        "sentiment":              MODELS_DIR / "sentiment"         / nom_local,
        "sentiment_multilingue":  MODELS_DIR / "sentiment"         / nom_local,
        "classification_zero_shot": MODELS_DIR / "classification"  / nom_local,
        "question_answering":     MODELS_DIR / "question_answering" / nom_local,
        "resume":                 MODELS_DIR / "summarization"     / nom_local,
        "ner":                    MODELS_DIR / "ner"               / nom_local,
        "embeddings_rag":         MODELS_DIR / "embeddings"        / nom_local,
    }
    return mapping.get(tache, MODELS_DIR / nom_local)


def afficher_barre_progression(current: int, total: int, nom: str, largeur: int = 40):
    """Affiche une barre de progression ASCII."""
    pct = current / total
    rempli = int(largeur * pct)
    barre = "█" * rempli + "░" * (largeur - rempli)
    print(f"\r  [{barre}] {pct:.0%}  {nom[:30]:<30}", end="", flush=True)


def sauvegarder_manifest(resultats: list):
    """Sauvegarde un fichier JSON de suivi des téléchargements."""
    manifest = {
        "date_telechargement": datetime.now().isoformat(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "modeles": resultats,
    }
    manifest_path = MODELS_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info(f"📄 Manifest sauvegardé : {manifest_path}")


# ════════════════════════════════════════════════════════════════════════════
# FONCTIONS DE TÉLÉCHARGEMENT PAR TYPE
# ════════════════════════════════════════════════════════════════════════════

def charger_auto(model_id: str, save_path: Path) -> bool:
    """Télécharge AutoTokenizer + AutoModel (BERT encodeur)."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    tokenizer.save_pretrained(str(save_path))
    model.save_pretrained(str(save_path))
    return True


def charger_classification(model_id: str, save_path: Path) -> bool:
    """Télécharge un modèle de classification de séquences."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    tokenizer.save_pretrained(str(save_path))
    model.save_pretrained(str(save_path))
    return True


def charger_question_answering(model_id: str, save_path: Path) -> bool:
    """Télécharge un modèle de Question Answering."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForQuestionAnswering.from_pretrained(model_id)
    tokenizer.save_pretrained(str(save_path))
    model.save_pretrained(str(save_path))
    return True


def charger_seq2seq(model_id: str, save_path: Path) -> bool:
    """Télécharge un modèle Seq2Seq (résumé, traduction)."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    tokenizer.save_pretrained(str(save_path))
    model.save_pretrained(str(save_path))
    return True


def charger_token_classification(model_id: str, save_path: Path) -> bool:
    """Télécharge un modèle de classification de tokens (NER)."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForTokenClassification.from_pretrained(model_id)
    tokenizer.save_pretrained(str(save_path))
    model.save_pretrained(str(save_path))
    return True


def charger_sentence_transformer(model_id: str, save_path: Path) -> bool:
    """Télécharge un Sentence Transformer."""
    model = SentenceTransformer(model_id)
    model.save(str(save_path))
    return True


# Dispatch par type
CHARGEURS = {
    "auto":                   charger_auto,
    "sequence_classification": charger_classification,
    "question_answering":     charger_question_answering,
    "seq2seq":                charger_seq2seq,
    "token_classification":   charger_token_classification,
    "sentence_transformer":   charger_sentence_transformer,
}


# ════════════════════════════════════════════════════════════════════════════
# FONCTION PRINCIPALE DE TÉLÉCHARGEMENT
# ════════════════════════════════════════════════════════════════════════════

def telecharger_modele(nom_local: str, config: dict) -> dict:
    """
    Télécharge et sauvegarde un modèle HuggingFace.

    Paramètres
    ----------
    nom_local : str   — clé locale du modèle
    config    : dict  — configuration (model_id, type, tache, ...)

    Retourne
    --------
    dict — résultat du téléchargement (status, durée, chemin)
    """
    model_id   = config["model_id"]
    model_type = config["type"]
    tache      = config["tache"]
    save_path  = get_save_path(nom_local, tache)

    resultat = {
        "nom_local": nom_local,
        "model_id":  model_id,
        "tache":     tache,
        "chemin":    str(save_path),
        "status":    None,
        "duree_s":   None,
        "erreur":    None,
    }

    # ── Vérifier si déjà téléchargé ──────────────────────────────────────
    if save_path.exists() and any(save_path.iterdir()):
        logger.info(f"⏭️  [{nom_local}] Déjà présent → {save_path}")
        resultat["status"] = "cache"
        return resultat

    # ── Téléchargement ────────────────────────────────────────────────────
    logger.info(f"\n{'─'*60}")
    logger.info(f"📥 Téléchargement : {nom_local}")
    logger.info(f"   Model ID  : {model_id}")
    logger.info(f"   Type      : {model_type}")
    logger.info(f"   Tâche     : {tache}")
    logger.info(f"   Taille    : {config.get('taille_approx', 'N/A')}")
    logger.info(f"   Desc.     : {config['description'][:80]}...")
    logger.info(f"   → Sauvegarde : {save_path}")

    save_path.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    try:
        chargeur = CHARGEURS.get(model_type)
        if chargeur is None:
            raise ValueError(f"Type inconnu : {model_type}")

        chargeur(model_id, save_path)
        duree = time.time() - t0

        # Sauvegarder la config du modèle
        meta = {**config, "nom_local": nom_local, "chemin_local": str(save_path),
                "date_telechargement": datetime.now().isoformat(), "duree_s": round(duree, 2)}
        with open(save_path / "model_info.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"✅ [{nom_local}] Téléchargé en {duree:.1f}s")
        resultat.update({"status": "ok", "duree_s": round(duree, 2)})

    except Exception as e:
        duree = time.time() - t0
        logger.error(f"❌ [{nom_local}] ERREUR après {duree:.1f}s : {e}")
        resultat.update({"status": "erreur", "duree_s": round(duree, 2), "erreur": str(e)})

    return resultat


# ════════════════════════════════════════════════════════════════════════════
# VÉRIFICATION DES MODÈLES CHARGÉS
# ════════════════════════════════════════════════════════════════════════════

def verifier_modeles():
    """
    Vérifie que tous les modèles peuvent être chargés depuis le disque.
    Lance une inférence rapide pour chaque modèle.
    """
    print("\n" + "═" * 60)
    print("  VÉRIFICATION DES MODÈLES")
    print("═" * 60)

    verifs = []

    # ── Tokenizer BERT ────────────────────────────────────────────────────
    try:
        path = MODELS_DIR / "tokenizers" / "bert-base-uncased"
        tok = AutoTokenizer.from_pretrained(str(path))
        ids = tok.encode("Test de vérification", add_special_tokens=True)
        print(f"  ✅ bert-base-uncased        → tokens: {len(ids)}")
        verifs.append(("bert-base-uncased", True))
    except Exception as e:
        print(f"  ❌ bert-base-uncased        → {e}")
        verifs.append(("bert-base-uncased", False))

    # ── Sentiment DistilBERT ──────────────────────────────────────────────
    try:
        path = MODELS_DIR / "sentiment" / "distilbert-sentiment"
        clf = pipeline("sentiment-analysis", model=str(path), tokenizer=str(path))
        r = clf("I love this!")[0]
        print(f"  ✅ distilbert-sentiment     → {r['label']} ({r['score']:.3f})")
        verifs.append(("distilbert-sentiment", True))
    except Exception as e:
        print(f"  ❌ distilbert-sentiment     → {e}")
        verifs.append(("distilbert-sentiment", False))

    # ── Question Answering ────────────────────────────────────────────────
    try:
        path = MODELS_DIR / "question_answering" / "roberta-squad2"
        qa = pipeline("question-answering", model=str(path), tokenizer=str(path))
        r = qa(question="What is AI?", context="AI is artificial intelligence.")
        print(f"  ✅ roberta-squad2           → '{r['answer']}'")
        verifs.append(("roberta-squad2", True))
    except Exception as e:
        print(f"  ❌ roberta-squad2           → {e}")
        verifs.append(("roberta-squad2", False))

    # ── Sentence Transformer ──────────────────────────────────────────────
    try:
        path = MODELS_DIR / "embeddings" / "all-MiniLM-L6-v2"
        model = SentenceTransformer(str(path))
        emb = model.encode(["test"])
        print(f"  ✅ all-MiniLM-L6-v2        → embedding shape: {emb.shape}")
        verifs.append(("all-MiniLM-L6-v2", True))
    except Exception as e:
        print(f"  ❌ all-MiniLM-L6-v2        → {e}")
        verifs.append(("all-MiniLM-L6-v2", False))

    # ── Résumé ────────────────────────────────────────────────────────────
    try:
        path = MODELS_DIR / "summarization" / "bart-large-cnn"
        summarizer = pipeline("summarization", model=str(path), tokenizer=str(path))
        r = summarizer("The quick brown fox. " * 10, max_length=20, min_length=5)
        print(f"  ✅ bart-large-cnn           → résumé OK ({len(r[0]['summary_text'].split())} mots)")
        verifs.append(("bart-large-cnn", True))
    except Exception as e:
        print(f"  ❌ bart-large-cnn           → {e}")
        verifs.append(("bart-large-cnn", False))

    # ── Résumé ────────────────────────────────────────────────────────────
    n_ok = sum(1 for _, ok in verifs if ok)
    print(f"\n  Résultat : {n_ok}/{len(verifs)} modèles vérifiés avec succès")
    return verifs


# ════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════

def main(modeles_specifiques: list = None, verifier: bool = True):
    """
    Télécharge tous les modèles configurés et les sauvegarde dans models/.

    Paramètres
    ----------
    modeles_specifiques : list ou None
        Si fourni, ne télécharge que ces modèles (par nom local).
        Ex: ["bert-base-uncased", "all-MiniLM-L6-v2"]
    verifier : bool
        Si True, vérifie les modèles après téléchargement.
    """

    print("╔" + "═" * 58 + "╗")
    print("║  MINI-PROJET NLP — TÉLÉCHARGEMENT DES MODÈLES             ║")
    print("║  ENSA Al Hoceima — Ingénierie des Données 2A              ║")
    print("╚" + "═" * 58 + "╝")
    print(f"\n  Device       : {'CUDA (GPU)' if torch.cuda.is_available() else 'CPU'}")
    print(f"  Dossier      : {MODELS_DIR}")
    print(f"  Nb modèles   : {len(MODELS_CONFIG)}")
    print()

    # Créer l'arborescence
    creer_arborescence()

    # Sélection des modèles
    modeles_a_charger = {
        k: v for k, v in MODELS_CONFIG.items()
        if modeles_specifiques is None or k in modeles_specifiques
    }

    print(f"\n  Modèles à télécharger ({len(modeles_a_charger)}) :")
    for i, (nom, cfg) in enumerate(modeles_a_charger.items(), 1):
        print(f"  {i:2}. {nom:<35} [{cfg['taille_approx']}]")

    print()
    t_global = time.time()
    resultats = []

    for i, (nom_local, config) in enumerate(modeles_a_charger.items(), 1):
        afficher_barre_progression(i - 1, len(modeles_a_charger), nom_local)
        resultat = telecharger_modele(nom_local, config)
        resultats.append(resultat)
        afficher_barre_progression(i, len(modeles_a_charger), nom_local)

    print()  # saut de ligne après barre de progression

    # Résumé
    duree_totale = time.time() - t_global
    n_ok     = sum(1 for r in resultats if r["status"] in ("ok", "cache"))
    n_cache  = sum(1 for r in resultats if r["status"] == "cache")
    n_erreur = sum(1 for r in resultats if r["status"] == "erreur")

    print("\n" + "═" * 60)
    print("  RÉSUMÉ DU TÉLÉCHARGEMENT")
    print("═" * 60)
    print(f"  ✅ Succès      : {n_ok}")
    print(f"  ⏭️  En cache   : {n_cache}")
    print(f"  ❌ Erreurs     : {n_erreur}")
    print(f"  ⏱️  Durée tot. : {duree_totale:.1f}s")

    for r in resultats:
        status_icon = {"ok": "✅", "cache": "⏭️ ", "erreur": "❌"}.get(r["status"], "?")
        duree = f"{r['duree_s']:.1f}s" if r["duree_s"] else "—"
        print(f"  {status_icon} {r['nom_local']:<35} {duree}")
        if r["erreur"]:
            print(f"       ↳ Erreur : {r['erreur']}")

    # Sauvegarde du manifest
    sauvegarder_manifest(resultats)

    # Vérification optionnelle
    if verifier:
        verifier_modeles()

    print("\n✅ Script terminé. Modèles disponibles dans :", MODELS_DIR)
    return resultats


# ════════════════════════════════════════════════════════════════════════════
# FONCTIONS HELPER — Chargement depuis le disque
# ════════════════════════════════════════════════════════════════════════════

def charger_depuis_disque(nom_local: str):
    """
    Charge un modèle déjà téléchargé depuis le dossier models/.

    Utilisation :
    >>> from src.exploration.load_models import charger_depuis_disque
    >>> model, tokenizer = charger_depuis_disque("distilbert-sentiment")

    Retourne : (model, tokenizer) ou SentenceTransformer
    """
    # Retrouver la config
    config = MODELS_CONFIG.get(nom_local)
    if config is None:
        raise ValueError(f"Modèle inconnu : {nom_local}. "
                         f"Options : {list(MODELS_CONFIG.keys())}")

    tache     = config["type"]
    save_path = get_save_path(nom_local, config["tache"])

    if not save_path.exists():
        raise FileNotFoundError(
            f"Modèle non trouvé : {save_path}\n"
            f"Lancez d'abord : python load_models.py"
        )

    logger.info(f"📂 Chargement depuis disque : {save_path}")

    if tache == "sentence_transformer":
        return SentenceTransformer(str(save_path))

    # Chargement tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(save_path))

    # Chargement modèle selon le type
    loaders = {
        "auto":                    AutoModel,
        "sequence_classification": AutoModelForSequenceClassification,
        "question_answering":      AutoModelForQuestionAnswering,
        "seq2seq":                 AutoModelForSeq2SeqLM,
        "token_classification":    AutoModelForTokenClassification,
    }
    model_class = loaders.get(tache, AutoModel)
    model = model_class.from_pretrained(str(save_path))

    return model, tokenizer


def lister_modeles_disponibles():
    """Affiche tous les modèles présents dans models/."""
    print("\n📂 MODÈLES DISPONIBLES DANS models/\n")
    print(f"  {'Nom local':<35} {'Tâche':<25} {'Taille'}")
    print("  " + "─" * 70)

    total_size = 0
    for nom, config in MODELS_CONFIG.items():
        save_path = get_save_path(nom, config["tache"])
        disponible = "✅" if (save_path.exists() and any(save_path.iterdir())) else "❌"

        # Calculer la taille sur disque
        if save_path.exists():
            size_bytes = sum(f.stat().st_size for f in save_path.rglob("*") if f.is_file())
            size_mb = size_bytes / (1024 * 1024)
            total_size += size_mb
            size_str = f"{size_mb:.0f} MB"
        else:
            size_str = "non téléchargé"

        print(f"  {disponible} {nom:<33} {config['tache']:<25} {size_str}")

    print(f"\n  Total sur disque : {total_size:.0f} MB ({total_size/1024:.1f} GB)")


# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Télécharge les modèles HuggingFace dans models/"
    )
    parser.add_argument(
        "--modeles", nargs="+", default=None,
        help="Noms locaux des modèles à télécharger (défaut: tous)"
    )
    parser.add_argument(
        "--no-verify", action="store_true",
        help="Ne pas vérifier les modèles après téléchargement"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Lister les modèles disponibles et quitter"
    )
    args = parser.parse_args()

    if args.list:
        lister_modeles_disponibles()
    else:
        main(
            modeles_specifiques=args.modeles,
            verifier=not args.no_verify,
        )