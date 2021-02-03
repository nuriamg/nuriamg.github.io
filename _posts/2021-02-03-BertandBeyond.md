---
layout: post
title: Paper Summary of "Pretrained Transformer for Text Ranking - Bert and Beyond"
---

### Introduction
**Text Ranking**: generate an ordered set of texts retrieved from a corpus in response to a query for a particular task. The most common form of text ranking is search.
* Examples: QA Question Answering, CQA Community QA, Information Filtering, Text Recommendation, Ranking as Input to Downstream Models.
* Examples beyond information access: Distant Supervision and Data Augmentation, Selecting from Competing Hypotheses.

**Ad Hoc Retrieval Problem**: we want to retrieve the relevant texts for a query that are about the topic of the user's request and address the need of information of the user.

**Exact Term Matching**: text from docs and texts from queries have to match exactly to contribute to the ranking.
    - It's a challenge when the query describes what it wants to search in different words as the documents. Then there is no exact match.

Deep Learning has freed the text ranking problem from the challenges of exact term matching.

Pre-BERT Models: 
* Representation-based: learn dense vector representation of queries and documents independently, and we compute the similarity between the representation with cosine similarity or inner product. DSSM, 2013. DESM, 2016.
* Interaction-based: compare representation of term of the query and documents directly and produce a similarity matrix. DRMM, 2016. KNRM, 2017.
* Both approaches can include neural networks for creating the representations.
* IB are more effective models but slower than RB.
* Hybrid Models: DUET, 2017.

BERT (Bidirectional Encoder Representations from Transformers, Devlin et al. 2019) is an algorithm that uses transformers (like LSTM but better). 
* Transformers were presented in 2017 by Vaswani et al. and BERT was presented in 2018. 
* Transformers are innovative in high level design choices and low-level implementation details.
* It has revolutionized the NLP and IR world.
* It was build in top of: Transformers (2017), ULMFiT (2018) → the idea of pre-training, ELMo (Embeddings from Language Models, 2018)→ ELMo used LSTM while BERT uses transformers but both use contextual embeddings, GPT (Generative Pre-trained Transformer, 2018).
* Transformers architecture + self-supervised pre-training.
* It is often incorporated as a part of a larger neural model.
* First use of BERT for text ranking: Nogueira and Cho 2019.


The easiest way to make your first post is to edit this one. Go into /_posts/ and update the Hello World markdown file. For more instructions head over to the [Jekyll Now repository](https://github.com/barryclark/jekyll-now) on GitHub.

----
****
