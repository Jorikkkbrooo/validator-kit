"""
DECLOUD Dataset Manager
=======================

Manages dataset downloads and validation for the DECLOUD network.
All datasets are mapped to the on-chain Dataset enum from lib.rs.
"""

import os
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class DatasetCategory(Enum):
    """Dataset categories matching lib.rs organization"""
    IMAGES_CLASSIFICATION = "images_classification"
    TEXT_SENTIMENT = "text_sentiment"
    TEXT_CLASSIFICATION = "text_classification"
    TEXT_SPAM_TOXICITY = "text_spam_toxicity"
    TEXT_INTENT = "text_intent"
    TEXT_NER = "text_ner"
    TEXT_QA = "text_qa"
    TEXT_SIMILARITY = "text_similarity"
    TEXT_SUMMARIZATION = "text_summarization"
    AUDIO_SPEECH = "audio_speech"
    AUDIO_MUSIC = "audio_music"
    AUDIO_EMOTION = "audio_emotion"
    TABULAR = "tabular"
    MEDICAL_IMAGES = "medical_images"
    MEDICAL_TEXT = "medical_text"
    TIME_SERIES = "time_series"
    CODE = "code"
    GRAPHS = "graphs"
    SECURITY = "security"
    RECOMMENDATION = "recommendation"
    MULTILINGUAL = "multilingual"
    CUSTOM = "custom"


@dataclass
class DatasetInfo:
    """Information about a dataset"""
    name: str                    # On-chain enum name (must match lib.rs exactly!)
    hf_path: str                 # HuggingFace path
    split: str                   # Which split to use
    category: DatasetCategory
    size_estimate: str           # Approximate size
    task: str                    # Task type


# ═══════════════════════════════════════════════════════════════════════════════
# DATASETS - Must match enum Dataset in lib.rs exactly!
# ═══════════════════════════════════════════════════════════════════════════════

DATASETS: Dict[str, DatasetInfo] = {
    # ───────────────────────────────────────────────────────────────────────────
    # IMAGES - CLASSIFICATION (16 datasets)
    # ───────────────────────────────────────────────────────────────────────────
    "Cifar10": DatasetInfo(
        name="Cifar10", hf_path="cifar10", split="test",
        category=DatasetCategory.IMAGES_CLASSIFICATION,
        size_estimate="170MB", task="image_classification"
    ),
    "Cifar100": DatasetInfo(
        name="Cifar100", hf_path="cifar100", split="test",
        category=DatasetCategory.IMAGES_CLASSIFICATION,
        size_estimate="170MB", task="image_classification"
    ),
    "Mnist": DatasetInfo(
        name="Mnist", hf_path="mnist", split="test",
        category=DatasetCategory.IMAGES_CLASSIFICATION,
        size_estimate="50MB", task="image_classification"
    ),
    "FashionMnist": DatasetInfo(
        name="FashionMnist", hf_path="fashion_mnist", split="test",
        category=DatasetCategory.IMAGES_CLASSIFICATION,
        size_estimate="50MB", task="image_classification"
    ),
    "Emnist": DatasetInfo(
        name="Emnist", hf_path="emnist", split="test",
        category=DatasetCategory.IMAGES_CLASSIFICATION,
        size_estimate="500MB", task="image_classification"
    ),
    "Kmnist": DatasetInfo(
        name="Kmnist", hf_path="kmnist", split="test",
        category=DatasetCategory.IMAGES_CLASSIFICATION,
        size_estimate="50MB", task="image_classification"
    ),
    "Food101": DatasetInfo(
        name="Food101", hf_path="food101", split="validation",
        category=DatasetCategory.IMAGES_CLASSIFICATION,
        size_estimate="5GB", task="image_classification"
    ),
    "Flowers102": DatasetInfo(
        name="Flowers102", hf_path="nelorth/oxford-flowers", split="test",
        category=DatasetCategory.IMAGES_CLASSIFICATION,
        size_estimate="500MB", task="image_classification"
    ),
    "StanfordDogs": DatasetInfo(
        name="StanfordDogs", hf_path="stanford_dogs", split="test",
        category=DatasetCategory.IMAGES_CLASSIFICATION,
        size_estimate="800MB", task="image_classification"
    ),
    "StanfordCars": DatasetInfo(
        name="StanfordCars", hf_path="stanford_cars", split="test",
        category=DatasetCategory.IMAGES_CLASSIFICATION,
        size_estimate="2GB", task="image_classification"
    ),
    "OxfordPets": DatasetInfo(
        name="OxfordPets", hf_path="oxford-iiit-pet", split="test",
        category=DatasetCategory.IMAGES_CLASSIFICATION,
        size_estimate="800MB", task="image_classification"
    ),
    "CatsVsDogs": DatasetInfo(
        name="CatsVsDogs", hf_path="cats_vs_dogs", split="train[:20%]",
        category=DatasetCategory.IMAGES_CLASSIFICATION,
        size_estimate="800MB", task="image_classification"
    ),
    "Eurosat": DatasetInfo(
        name="Eurosat", hf_path="eurosat", split="test",
        category=DatasetCategory.IMAGES_CLASSIFICATION,
        size_estimate="2GB", task="image_classification"
    ),
    "Svhn": DatasetInfo(
        name="Svhn", hf_path="svhn", split="test",
        category=DatasetCategory.IMAGES_CLASSIFICATION,
        size_estimate="500MB", task="image_classification"
    ),
    "Caltech101": DatasetInfo(
        name="Caltech101", hf_path="caltech101", split="test",
        category=DatasetCategory.IMAGES_CLASSIFICATION,
        size_estimate="150MB", task="image_classification"
    ),
    "Caltech256": DatasetInfo(
        name="Caltech256", hf_path="caltech256", split="test",
        category=DatasetCategory.IMAGES_CLASSIFICATION,
        size_estimate="1GB", task="image_classification"
    ),

    # ───────────────────────────────────────────────────────────────────────────
    # TEXT - SENTIMENT (8 datasets)
    # ───────────────────────────────────────────────────────────────────────────
    "Imdb": DatasetInfo(
        name="Imdb", hf_path="imdb", split="test",
        category=DatasetCategory.TEXT_SENTIMENT,
        size_estimate="100MB", task="text_classification"
    ),
    "Sst2": DatasetInfo(
        name="Sst2", hf_path="sst2", split="validation",
        category=DatasetCategory.TEXT_SENTIMENT,
        size_estimate="10MB", task="text_classification"
    ),
    "Sst5": DatasetInfo(
        name="Sst5", hf_path="SetFit/sst5", split="test",
        category=DatasetCategory.TEXT_SENTIMENT,
        size_estimate="10MB", task="text_classification"
    ),
    "YelpReviews": DatasetInfo(
        name="YelpReviews", hf_path="yelp_review_full", split="test",
        category=DatasetCategory.TEXT_SENTIMENT,
        size_estimate="500MB", task="text_classification"
    ),
    "AmazonPolarity": DatasetInfo(
        name="AmazonPolarity", hf_path="amazon_polarity", split="test",
        category=DatasetCategory.TEXT_SENTIMENT,
        size_estimate="1GB", task="text_classification"
    ),
    "RottenTomatoes": DatasetInfo(
        name="RottenTomatoes", hf_path="rotten_tomatoes", split="test",
        category=DatasetCategory.TEXT_SENTIMENT,
        size_estimate="5MB", task="text_classification"
    ),
    "FinancialSentiment": DatasetInfo(
        name="FinancialSentiment", hf_path="financial_phrasebank", split="train",
        category=DatasetCategory.TEXT_SENTIMENT,
        size_estimate="1MB", task="text_classification"
    ),
    "TweetSentiment": DatasetInfo(
        name="TweetSentiment", hf_path="mteb/tweet_sentiment_extraction", split="test",
        category=DatasetCategory.TEXT_SENTIMENT,
        size_estimate="5MB", task="text_classification"
    ),

    # ───────────────────────────────────────────────────────────────────────────
    # TEXT - CLASSIFICATION (4 datasets)
    # ───────────────────────────────────────────────────────────────────────────
    "AgNews": DatasetInfo(
        name="AgNews", hf_path="ag_news", split="test",
        category=DatasetCategory.TEXT_CLASSIFICATION,
        size_estimate="30MB", task="text_classification"
    ),
    "Dbpedia": DatasetInfo(
        name="Dbpedia", hf_path="dbpedia_14", split="test",
        category=DatasetCategory.TEXT_CLASSIFICATION,
        size_estimate="100MB", task="text_classification"
    ),
    "YahooAnswers": DatasetInfo(
        name="YahooAnswers", hf_path="yahoo_answers_topics", split="test",
        category=DatasetCategory.TEXT_CLASSIFICATION,
        size_estimate="500MB", task="text_classification"
    ),
    "TwentyNewsgroups": DatasetInfo(
        name="TwentyNewsgroups", hf_path="SetFit/20_newsgroups", split="test",
        category=DatasetCategory.TEXT_CLASSIFICATION,
        size_estimate="20MB", task="text_classification"
    ),

    # ───────────────────────────────────────────────────────────────────────────
    # TEXT - SPAM & TOXICITY (4 datasets)
    # ───────────────────────────────────────────────────────────────────────────
    "SmsSpam": DatasetInfo(
        name="SmsSpam", hf_path="sms_spam", split="train",
        category=DatasetCategory.TEXT_SPAM_TOXICITY,
        size_estimate="1MB", task="text_classification"
    ),
    "HateSpeech": DatasetInfo(
        name="HateSpeech", hf_path="hate_speech18", split="train",
        category=DatasetCategory.TEXT_SPAM_TOXICITY,
        size_estimate="5MB", task="text_classification"
    ),
    "CivilComments": DatasetInfo(
        name="CivilComments", hf_path="civil_comments", split="test",
        category=DatasetCategory.TEXT_SPAM_TOXICITY,
        size_estimate="500MB", task="text_classification"
    ),
    "Toxicity": DatasetInfo(
        name="Toxicity", hf_path="OxAISH-AL-LLM/wiki_toxic", split="test",
        category=DatasetCategory.TEXT_SPAM_TOXICITY,
        size_estimate="100MB", task="text_classification"
    ),

    # ───────────────────────────────────────────────────────────────────────────
    # TEXT - INTENT (3 datasets)
    # ───────────────────────────────────────────────────────────────────────────
    "ClincIntent": DatasetInfo(
        name="ClincIntent", hf_path="clinc_oos", split="test",
        category=DatasetCategory.TEXT_INTENT,
        size_estimate="5MB", task="text_classification"
    ),
    "Banking77": DatasetInfo(
        name="Banking77", hf_path="banking77", split="test",
        category=DatasetCategory.TEXT_INTENT,
        size_estimate="2MB", task="text_classification"
    ),
    "SnipsIntent": DatasetInfo(
        name="SnipsIntent", hf_path="snips_built_in_intents", split="test",
        category=DatasetCategory.TEXT_INTENT,
        size_estimate="1MB", task="text_classification"
    ),

    # ───────────────────────────────────────────────────────────────────────────
    # TEXT - NER (2 datasets)
    # ───────────────────────────────────────────────────────────────────────────
    "Conll2003": DatasetInfo(
        name="Conll2003", hf_path="conll2003", split="test",
        category=DatasetCategory.TEXT_NER,
        size_estimate="10MB", task="token_classification"
    ),
    "Wnut17": DatasetInfo(
        name="Wnut17", hf_path="wnut_17", split="test",
        category=DatasetCategory.TEXT_NER,
        size_estimate="5MB", task="token_classification"
    ),

    # ───────────────────────────────────────────────────────────────────────────
    # TEXT - QA (5 datasets)
    # ───────────────────────────────────────────────────────────────────────────
    "Squad": DatasetInfo(
        name="Squad", hf_path="squad", split="validation",
        category=DatasetCategory.TEXT_QA,
        size_estimate="50MB", task="question_answering"
    ),
    "SquadV2": DatasetInfo(
        name="SquadV2", hf_path="squad_v2", split="validation",
        category=DatasetCategory.TEXT_QA,
        size_estimate="50MB", task="question_answering"
    ),
    "TriviaQa": DatasetInfo(
        name="TriviaQa", hf_path="trivia_qa", split="validation",
        category=DatasetCategory.TEXT_QA,
        size_estimate="2GB", task="question_answering"
    ),
    "BoolQ": DatasetInfo(
        name="BoolQ", hf_path="boolq", split="validation",
        category=DatasetCategory.TEXT_QA,
        size_estimate="10MB", task="question_answering"
    ),
    "CommonsenseQa": DatasetInfo(
        name="CommonsenseQa", hf_path="commonsense_qa", split="validation",
        category=DatasetCategory.TEXT_QA,
        size_estimate="5MB", task="question_answering"
    ),

    # ───────────────────────────────────────────────────────────────────────────
    # TEXT - SIMILARITY (5 datasets)
    # ───────────────────────────────────────────────────────────────────────────
    "Stsb": DatasetInfo(
        name="Stsb", hf_path="mteb/stsbenchmark-sts", split="test",
        category=DatasetCategory.TEXT_SIMILARITY,
        size_estimate="1MB", task="text_similarity"
    ),
    "Mrpc": DatasetInfo(
        name="Mrpc", hf_path="glue", split="test",
        category=DatasetCategory.TEXT_SIMILARITY,
        size_estimate="2MB", task="text_similarity"
    ),
    "Qqp": DatasetInfo(
        name="Qqp", hf_path="glue", split="validation",
        category=DatasetCategory.TEXT_SIMILARITY,
        size_estimate="50MB", task="text_similarity"
    ),
    "Snli": DatasetInfo(
        name="Snli", hf_path="snli", split="test",
        category=DatasetCategory.TEXT_SIMILARITY,
        size_estimate="100MB", task="text_similarity"
    ),
    "Mnli": DatasetInfo(
        name="Mnli", hf_path="multi_nli", split="validation_matched",
        category=DatasetCategory.TEXT_SIMILARITY,
        size_estimate="100MB", task="text_similarity"
    ),

    # ───────────────────────────────────────────────────────────────────────────
    # TEXT - SUMMARIZATION (3 datasets)
    # ───────────────────────────────────────────────────────────────────────────
    "CnnDailymail": DatasetInfo(
        name="CnnDailymail", hf_path="cnn_dailymail", split="test",
        category=DatasetCategory.TEXT_SUMMARIZATION,
        size_estimate="1GB", task="summarization"
    ),
    "Xsum": DatasetInfo(
        name="Xsum", hf_path="xsum", split="test",
        category=DatasetCategory.TEXT_SUMMARIZATION,
        size_estimate="500MB", task="summarization"
    ),
    "Samsum": DatasetInfo(
        name="Samsum", hf_path="samsum", split="test",
        category=DatasetCategory.TEXT_SUMMARIZATION,
        size_estimate="10MB", task="summarization"
    ),

    # ───────────────────────────────────────────────────────────────────────────
    # AUDIO - SPEECH (3 datasets)
    # ───────────────────────────────────────────────────────────────────────────
    "SpeechCommands": DatasetInfo(
        name="SpeechCommands", hf_path="speech_commands", split="test",
        category=DatasetCategory.AUDIO_SPEECH,
        size_estimate="2GB", task="audio_classification"
    ),
    "Librispeech": DatasetInfo(
        name="Librispeech", hf_path="librispeech_asr", split="test.clean",
        category=DatasetCategory.AUDIO_SPEECH,
        size_estimate="10GB", task="speech_recognition"
    ),
    "CommonVoice": DatasetInfo(
        name="CommonVoice", hf_path="mozilla-foundation/common_voice_11_0", split="test",
        category=DatasetCategory.AUDIO_SPEECH,
        size_estimate="50GB", task="speech_recognition"
    ),

    # ───────────────────────────────────────────────────────────────────────────
    # AUDIO - MUSIC (4 datasets)
    # ───────────────────────────────────────────────────────────────────────────
    "Gtzan": DatasetInfo(
        name="Gtzan", hf_path="marsyas/gtzan", split="train",
        category=DatasetCategory.AUDIO_MUSIC,
        size_estimate="1GB", task="audio_classification"
    ),
    "Esc50": DatasetInfo(
        name="Esc50", hf_path="ashraq/esc50", split="test",
        category=DatasetCategory.AUDIO_MUSIC,
        size_estimate="600MB", task="audio_classification"
    ),
    "Urbansound8k": DatasetInfo(
        name="Urbansound8k", hf_path="danavery/urbansound8K", split="train",
        category=DatasetCategory.AUDIO_MUSIC,
        size_estimate="6GB", task="audio_classification"
    ),
    "Nsynth": DatasetInfo(
        name="Nsynth", hf_path="nsynth", split="test",
        category=DatasetCategory.AUDIO_MUSIC,
        size_estimate="20GB", task="audio_classification"
    ),

    # ───────────────────────────────────────────────────────────────────────────
    # AUDIO - EMOTION (3 datasets)
    # ───────────────────────────────────────────────────────────────────────────
    "Ravdess": DatasetInfo(
        name="Ravdess", hf_path="narad/ravdess", split="train",
        category=DatasetCategory.AUDIO_EMOTION,
        size_estimate="1GB", task="audio_classification"
    ),
    "CremaD": DatasetInfo(
        name="CremaD", hf_path="flexthink/crema-d", split="train",
        category=DatasetCategory.AUDIO_EMOTION,
        size_estimate="2GB", task="audio_classification"
    ),
    "Iemocap": DatasetInfo(
        name="Iemocap", hf_path="Zahra99/IEMOCAP_Audio", split="train",
        category=DatasetCategory.AUDIO_EMOTION,
        size_estimate="5GB", task="audio_classification"
    ),

    # ───────────────────────────────────────────────────────────────────────────
    # TABULAR (10 datasets)
    # ───────────────────────────────────────────────────────────────────────────
    "Iris": DatasetInfo(
        name="Iris", hf_path="scikit-learn/iris", split="train",
        category=DatasetCategory.TABULAR,
        size_estimate="10KB", task="tabular_classification"
    ),
    "Wine": DatasetInfo(
        name="Wine", hf_path="scikit-learn/wine-quality", split="train",
        category=DatasetCategory.TABULAR,
        size_estimate="100KB", task="tabular_classification"
    ),
    "Diabetes": DatasetInfo(
        name="Diabetes", hf_path="scikit-learn/diabetes", split="train",
        category=DatasetCategory.TABULAR,
        size_estimate="50KB", task="tabular_regression"
    ),
    "BreastCancer": DatasetInfo(
        name="BreastCancer", hf_path="scikit-learn/breast-cancer", split="train",
        category=DatasetCategory.TABULAR,
        size_estimate="50KB", task="tabular_classification"
    ),
    "CaliforniaHousing": DatasetInfo(
        name="CaliforniaHousing", hf_path="scikit-learn/california-housing", split="train",
        category=DatasetCategory.TABULAR,
        size_estimate="500KB", task="tabular_regression"
    ),
    "AdultIncome": DatasetInfo(
        name="AdultIncome", hf_path="scikit-learn/adult-census-income", split="train",
        category=DatasetCategory.TABULAR,
        size_estimate="5MB", task="tabular_classification"
    ),
    "BankMarketing": DatasetInfo(
        name="BankMarketing", hf_path="scikit-learn/banking", split="train",
        category=DatasetCategory.TABULAR,
        size_estimate="5MB", task="tabular_classification"
    ),
    "CreditDefault": DatasetInfo(
        name="CreditDefault", hf_path="imodels/credit-card", split="test",
        category=DatasetCategory.TABULAR,
        size_estimate="3MB", task="tabular_classification"
    ),
    "Titanic": DatasetInfo(
        name="Titanic", hf_path="phihung/titanic", split="train",
        category=DatasetCategory.TABULAR,
        size_estimate="100KB", task="tabular_classification"
    ),
    "HeartDisease": DatasetInfo(
        name="HeartDisease", hf_path="codesignal/heart-disease", split="train",
        category=DatasetCategory.TABULAR,
        size_estimate="50KB", task="tabular_classification"
    ),

    # ───────────────────────────────────────────────────────────────────────────
    # MEDICAL - IMAGES (7 datasets)
    # ───────────────────────────────────────────────────────────────────────────
    "ChestXray": DatasetInfo(
        name="ChestXray", hf_path="keremberke/chest-xray-classification", split="test",
        category=DatasetCategory.MEDICAL_IMAGES,
        size_estimate="500MB", task="image_classification"
    ),
    "SkinCancer": DatasetInfo(
        name="SkinCancer", hf_path="marmal88/skin_cancer", split="test",
        category=DatasetCategory.MEDICAL_IMAGES,
        size_estimate="3GB", task="image_classification"
    ),
    "DiabeticRetinopathy": DatasetInfo(
        name="DiabeticRetinopathy", hf_path="aharley/diabetic_retinopathy", split="test",
        category=DatasetCategory.MEDICAL_IMAGES,
        size_estimate="10GB", task="image_classification"
    ),
    "BrainTumor": DatasetInfo(
        name="BrainTumor", hf_path="sartajbhuvaji/brain-tumor-classification", split="test",
        category=DatasetCategory.MEDICAL_IMAGES,
        size_estimate="200MB", task="image_classification"
    ),
    "Malaria": DatasetInfo(
        name="Malaria", hf_path="lhoestq/malaria", split="test",
        category=DatasetCategory.MEDICAL_IMAGES,
        size_estimate="500MB", task="image_classification"
    ),
    "BloodCells": DatasetInfo(
        name="BloodCells", hf_path="Falah/Blood_Cells_Cancer", split="test",
        category=DatasetCategory.MEDICAL_IMAGES,
        size_estimate="300MB", task="image_classification"
    ),
    "CovidXray": DatasetInfo(
        name="CovidXray", hf_path="keremberke/covid-xray-classification", split="test",
        category=DatasetCategory.MEDICAL_IMAGES,
        size_estimate="200MB", task="image_classification"
    ),

    # ───────────────────────────────────────────────────────────────────────────
    # MEDICAL - TEXT (2 datasets)
    # ───────────────────────────────────────────────────────────────────────────
    "PubmedQa": DatasetInfo(
        name="PubmedQa", hf_path="pubmed_qa", split="pqa_labeled",
        category=DatasetCategory.MEDICAL_TEXT,
        size_estimate="50MB", task="question_answering"
    ),
    "MedQa": DatasetInfo(
        name="MedQa", hf_path="bigbio/med_qa", split="test",
        category=DatasetCategory.MEDICAL_TEXT,
        size_estimate="20MB", task="question_answering"
    ),

    # ───────────────────────────────────────────────────────────────────────────
    # TIME SERIES (4 datasets)
    # ───────────────────────────────────────────────────────────────────────────
    "Electricity": DatasetInfo(
        name="Electricity", hf_path="monash_tsf", split="test",
        category=DatasetCategory.TIME_SERIES,
        size_estimate="500MB", task="time_series_forecasting"
    ),
    "Weather": DatasetInfo(
        name="Weather", hf_path="weather_bench", split="test",
        category=DatasetCategory.TIME_SERIES,
        size_estimate="1GB", task="time_series_forecasting"
    ),
    "StockPrices": DatasetInfo(
        name="StockPrices", hf_path="edarchimbaud/timeseries-1d-stocks", split="train",
        category=DatasetCategory.TIME_SERIES,
        size_estimate="100MB", task="time_series_forecasting"
    ),
    "EcgHeartbeat": DatasetInfo(
        name="EcgHeartbeat", hf_path="ceyda/ecg-heartbeat-categorization", split="test",
        category=DatasetCategory.TIME_SERIES,
        size_estimate="200MB", task="time_series_classification"
    ),

    # ───────────────────────────────────────────────────────────────────────────
    # CODE (4 datasets)
    # ───────────────────────────────────────────────────────────────────────────
    "CodeSearchNet": DatasetInfo(
        name="CodeSearchNet", hf_path="code_search_net", split="test",
        category=DatasetCategory.CODE,
        size_estimate="20GB", task="code_generation"
    ),
    "Humaneval": DatasetInfo(
        name="Humaneval", hf_path="openai_humaneval", split="test",
        category=DatasetCategory.CODE,
        size_estimate="50KB", task="code_generation"
    ),
    "Mbpp": DatasetInfo(
        name="Mbpp", hf_path="mbpp", split="test",
        category=DatasetCategory.CODE,
        size_estimate="1MB", task="code_generation"
    ),
    "Spider": DatasetInfo(
        name="Spider", hf_path="spider", split="validation",
        category=DatasetCategory.CODE,
        size_estimate="100MB", task="text_to_sql"
    ),

    # ───────────────────────────────────────────────────────────────────────────
    # GRAPHS (3 datasets)
    # ───────────────────────────────────────────────────────────────────────────
    "Cora": DatasetInfo(
        name="Cora", hf_path="graphs-datasets/cora", split="train",
        category=DatasetCategory.GRAPHS,
        size_estimate="10MB", task="node_classification"
    ),
    "Citeseer": DatasetInfo(
        name="Citeseer", hf_path="graphs-datasets/citeseer", split="train",
        category=DatasetCategory.GRAPHS,
        size_estimate="10MB", task="node_classification"
    ),
    "Qm9": DatasetInfo(
        name="Qm9", hf_path="graphs-datasets/qm9", split="train",
        category=DatasetCategory.GRAPHS,
        size_estimate="500MB", task="graph_regression"
    ),

    # ───────────────────────────────────────────────────────────────────────────
    # SECURITY (3 datasets)
    # ───────────────────────────────────────────────────────────────────────────
    "NslKdd": DatasetInfo(
        name="NslKdd", hf_path="CIS-MSIT-IDAE/nsl-kdd", split="test",
        category=DatasetCategory.SECURITY,
        size_estimate="20MB", task="tabular_classification"
    ),
    "CreditCardFraud": DatasetInfo(
        name="CreditCardFraud", hf_path="nelgiriyewithana/credit-card-fraud-detection-dataset-2023", split="train",
        category=DatasetCategory.SECURITY,
        size_estimate="150MB", task="tabular_classification"
    ),
    "Phishing": DatasetInfo(
        name="Phishing", hf_path="pirocheto/phishing-detection", split="test",
        category=DatasetCategory.SECURITY,
        size_estimate="10MB", task="tabular_classification"
    ),

    # ───────────────────────────────────────────────────────────────────────────
    # RECOMMENDATION (2 datasets)
    # ───────────────────────────────────────────────────────────────────────────
    "Movielens1m": DatasetInfo(
        name="Movielens1m", hf_path="reczilla/movielens-1m", split="test",
        category=DatasetCategory.RECOMMENDATION,
        size_estimate="25MB", task="recommendation"
    ),
    "Movielens100k": DatasetInfo(
        name="Movielens100k", hf_path="reczilla/movielens-100k", split="test",
        category=DatasetCategory.RECOMMENDATION,
        size_estimate="5MB", task="recommendation"
    ),

    # ───────────────────────────────────────────────────────────────────────────
    # MULTILINGUAL (3 datasets)
    # ───────────────────────────────────────────────────────────────────────────
    "Xnli": DatasetInfo(
        name="Xnli", hf_path="xnli", split="test",
        category=DatasetCategory.MULTILINGUAL,
        size_estimate="50MB", task="text_classification"
    ),
    "AmazonReviewsMulti": DatasetInfo(
        name="AmazonReviewsMulti", hf_path="amazon_reviews_multi", split="test",
        category=DatasetCategory.MULTILINGUAL,
        size_estimate="500MB", task="text_classification"
    ),
    "Sberquad": DatasetInfo(
        name="Sberquad", hf_path="sberquad", split="test",
        category=DatasetCategory.MULTILINGUAL,
        size_estimate="50MB", task="question_answering"
    ),

    # ───────────────────────────────────────────────────────────────────────────
    # CUSTOM
    # ───────────────────────────────────────────────────────────────────────────
    "Custom": DatasetInfo(
        name="Custom", hf_path="", split="",
        category=DatasetCategory.CUSTOM,
        size_estimate="varies", task="custom"
    ),
}


class DatasetManager:
    """
    Manages dataset downloads and access for DECLOUD validators.
    """
    
    def __init__(self, data_dir: str = "./decloud_datasets"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        if not HF_AVAILABLE:
            print("⚠️  Warning: 'datasets' library not installed")
            print("   Install with: pip install datasets")
    
    def list_datasets(self) -> List[str]:
        return list(DATASETS.keys())
    
    def list_categories(self) -> Dict[str, List[str]]:
        result = {}
        for name, info in DATASETS.items():
            cat = info.category.value
            if cat not in result:
                result[cat] = []
            result[cat].append(name)
        return result
    
    def get_info(self, name: str) -> Optional[DatasetInfo]:
        return DATASETS.get(name)
    
    def is_downloaded(self, name: str) -> bool:
        """Check if dataset is downloaded (supports both HF and torchvision formats)"""
        # Normalize name
        name_lower = name.lower().replace("_", "")
        
        # Check HuggingFace format: ./data/Cifar10
        hf_path = os.path.join(self.data_dir, name)
        if os.path.exists(hf_path):
            return True
        
        # Check torchvision formats
        torchvision_paths = {
            "cifar10": ["cifar-10-batches-py", "cifar-10-python.tar.gz"],
            "cifar100": ["cifar-100-python", "cifar-100-python.tar.gz"],
            "mnist": ["MNIST", "MNIST/raw"],
            "fashionmnist": ["FashionMNIST", "FashionMNIST/raw"],
            "emnist": ["EMNIST", "EMNIST/raw"],
            "kmnist": ["KMNIST", "KMNIST/raw"],
            "svhn": ["svhn", "train_32x32.mat"],
        }
        
        if name_lower in torchvision_paths:
            for subpath in torchvision_paths[name_lower]:
                full_path = os.path.join(self.data_dir, subpath)
                if os.path.exists(full_path):
                    return True
        
        return False
    
    def download(self, name: str, force: bool = False) -> bool:
        if not HF_AVAILABLE:
            print("❌ Cannot download: 'datasets' library not installed")
            return False
        
        info = DATASETS.get(name)
        if not info:
            print(f"❌ Unknown dataset: {name}")
            return False
        
        if info.name == "Custom":
            return True
        
        save_path = os.path.join(self.data_dir, name)
        
        if os.path.exists(save_path) and not force:
            print(f"✓ {name} already exists")
            return True
        
        print(f"📥 Downloading {name} ({info.size_estimate})...")
        
        try:
            if info.hf_path == "glue":
                ds = load_dataset(info.hf_path, name.lower(), split=info.split, trust_remote_code=True)
            elif info.hf_path == "cnn_dailymail":
                ds = load_dataset(info.hf_path, "3.0.0", split=info.split, trust_remote_code=True)
            elif info.hf_path == "trivia_qa":
                ds = load_dataset(info.hf_path, "rc", split=info.split, trust_remote_code=True)
            elif info.hf_path == "emnist":
                ds = load_dataset(info.hf_path, "balanced", split=info.split, trust_remote_code=True)
            elif info.hf_path == "svhn":
                ds = load_dataset(info.hf_path, "cropped_digits", split=info.split, trust_remote_code=True)
            else:
                ds = load_dataset(info.hf_path, split=info.split, trust_remote_code=True)
            
            ds.save_to_disk(save_path)
            print(f"✓ {name} downloaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to download {name}: {str(e)[:100]}")
            return False
    
    def download_category(self, category: DatasetCategory, force: bool = False) -> Dict[str, bool]:
        results = {}
        for name, info in DATASETS.items():
            if info.category == category:
                results[name] = self.download(name, force)
        return results
    
    def download_all(self, skip_large: bool = True, force: bool = False) -> Dict[str, bool]:
        large_datasets = ["Librispeech", "CommonVoice", "DiabeticRetinopathy", 
                         "CodeSearchNet", "Nsynth", "Urbansound8k", "Food101"]
        
        results = {}
        total = len(DATASETS)
        
        for i, (name, info) in enumerate(DATASETS.items(), 1):
            print(f"\n[{i}/{total}] Processing {name}...")
            
            if skip_large and name in large_datasets:
                print(f"⏭️  Skipping {name} (too large)")
                results[name] = False
                continue
            
            if info.name == "Custom":
                continue
                
            results[name] = self.download(name, force)
        
        success = sum(1 for v in results.values() if v)
        failed = sum(1 for v in results.values() if not v)
        
        print(f"\n{'='*60}")
        print(f"📊 Download Summary: ✓ {success} | ❌ {failed}")
        print(f"📁 Data dir: {self.data_dir}")
        
        return results
    
    def download_minimal(self) -> Dict[str, bool]:
        """Download minimal set for testing"""
        minimal = ["Cifar10", "Mnist", "FashionMnist", "Imdb", "Sst2", "AgNews", "Iris", "Wine", "Titanic"]
        results = {}
        for name in minimal:
            results[name] = self.download(name)
        return results
    
    def load(self, name: str):
        if not HF_AVAILABLE:
            raise RuntimeError("'datasets' library not installed")
        
        from datasets import load_from_disk
        save_path = os.path.join(self.data_dir, name)
        
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"Dataset {name} not downloaded")
        
        return load_from_disk(save_path)
    
    def estimate_total_size(self) -> str:
        return "~50 GB (standard), ~500 MB (minimal)"
    
    def status(self) -> Dict[str, Dict]:
        status = {}
        for name, info in DATASETS.items():
            status[name] = {
                "downloaded": self.is_downloaded(name),
                "category": info.category.value,
                "size": info.size_estimate,
                "task": info.task,
            }
        return status
    
    def print_status(self):
        print(f"\n{'='*70}")
        print("📊 DECLOUD Dataset Status (matching lib.rs enum)")
        print(f"{'='*70}")
        
        categories = self.list_categories()
        
        for cat_name, datasets in categories.items():
            downloaded = sum(1 for d in datasets if self.is_downloaded(d))
            total = len(datasets)
            status = "✓" if downloaded == total else f"{downloaded}/{total}"
            print(f"\n{cat_name}: {status}")
            
            for name in datasets:
                info = DATASETS[name]
                icon = "✓" if self.is_downloaded(name) else "○"
                print(f"  {icon} {name} ({info.size_estimate})")
        
        print(f"\n📁 Data directory: {self.data_dir}")
        print(f"📦 Total datasets: {len(DATASETS)} (matching on-chain enum)")