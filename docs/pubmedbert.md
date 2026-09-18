# PubMedBERT

[PubMedBERT](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract) is a family of BERT-style language models pretrained by Microsoft Research on biomedical literature. The models were introduced in Gu et al., *Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing* (ACM Transactions on Computing for Healthcare, 2021). On Hugging Face they are published under the `microsoft/BiomedNLP-...` namespace. The checkpoints were later renamed from "PubMedBERT" to "BiomedBERT", so you will see both names.

## Key Idea: Pretrain From Scratch

Earlier biomedical models such as BioBERT started from a general-domain BERT checkpoint and continued pretraining on PubMed text. The PubMedBERT authors argued this is suboptimal for a domain with very different vocabulary. Their approach:

- **Pretrain from scratch** on biomedical text only, with no general-domain text (Wikipedia, books) in the corpus.
- **Build a domain-specific WordPiece vocabulary** from PubMed text. General BERT vocabularies split terms like `acetyltransferase` or `brachytherapy` into many meaningless subwords. A biomedical vocabulary keeps many such terms whole or splits them into meaningful pieces.

This gave better results than continual pretraining from general BERT across most biomedical tasks.

## Model Variants

| Model | Pretraining corpus | Size |
|---|---|---|
| `BiomedNLP-BiomedBERT-base-uncased-abstract` | PubMed abstracts (about 14M abstracts, roughly 3.2B words) | Base, about 110M parameters |
| `BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` | PubMed abstracts plus PubMed Central full-text articles (roughly 16.8B words) | Base, about 110M parameters |
| `BiomedNLP-BiomedBERT-large-uncased-abstract` | PubMed abstracts | Large, about 335M parameters |

Common properties:

- Architecture: standard BERT encoder (12 layers and 768 hidden units for base; 24 layers and 1024 hidden units for large)
- Uncased WordPiece tokenizer with a vocabulary of roughly 30k tokens
- Maximum input length of 512 tokens
- Objective: masked language modeling
- Encoder-only, so suited to classification, extraction, and embeddings rather than text generation

## BLURB Benchmark

The same paper introduced **BLURB** (Biomedical Language Understanding and Reasoning Benchmark), which covers 13 datasets across 6 task types:

- Named entity recognition
- Relation extraction
- PICO extraction
- Sentence similarity
- Document classification
- Question answering

PubMedBERT set state-of-the-art results on BLURB at publication, outperforming BioBERT, SciBERT, and ClinicalBERT. It is still a common baseline for biomedical NLP.

## Derived and Related Models

Many downstream models use PubMedBERT as their starting point:

- **SapBERT** (Cambridge LTL): PubMedBERT fine-tuned for biomedical entity linking (UMLS concepts)
- **BiomedCLIP** (Microsoft): uses PubMedBERT as the text encoder in an image-text model
- **PubMedBERT sentence-embedding models** (for example, `NeuML/pubmedbert-base-embeddings` and MS MARCO-tuned variants): PubMedBERT fine-tuned for semantic search and similarity
- **Task fine-tunes**: community checkpoints for NER, relation extraction, and classification

The family should not be confused with generative biomedical models such as BioGPT or BioMedLM, which are decoder-only.

## Loading a Model

```python
from transformers import AutoTokenizer, AutoModel

name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModel.from_pretrained(name)

inputs = tokenizer("Brachytherapy for low-risk prostate cancer",
                   return_tensors="pt", truncation=True, max_length=512)
outputs = model(**inputs)
cls_embedding = outputs.last_hidden_state[:, 0]  # 768-dim vector
```

The raw `[CLS]` vector from the base checkpoint is not tuned for similarity. For paper-to-paper similarity, a sentence-embedding fine-tune usually works better.

## Relevance to This Project

The current pipeline uses regex-based scoring (`src/prostate-cancer-scorer.py`) and TF-IDF plus UMAP for the similarity visualization (`src/create-embedding/create-embeddings.py`). PubMedBERT could be used in several ways:

1. **Better embeddings**: replace TF-IDF with PubMedBERT-based abstract embeddings, so papers with similar meaning cluster together even when they share few words.
2. **Treatment classification**: fine-tune a PubMedBERT classifier to identify the 17 treatment modalities. This would catch synonyms and phrasing that regexes miss.
3. **Entity extraction**: fine-tune for NER on endpoints (BRFS, OS, MFS, CSS), risk stratification schemes, and numeric values such as dose and follow-up.

Supervised uses (2 and 3) need labeled data. The existing regex scores could serve as weak labels, but they should be checked against a hand-labeled sample first.

## References

- Gu et al., [Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing](https://arxiv.org/abs/2007.15779), 2020
- [BLURB leaderboard](https://microsoft.github.io/BLURB/)
- [Hugging Face: BiomedBERT base, abstract](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract)
- [Hugging Face: BiomedBERT base, abstract and full text](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext)
- [Hugging Face: BiomedBERT large, abstract](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract)
