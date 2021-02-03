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

### Setting the Stage
**Evaluation paradigms**: Cranfield Paradigm, A/B testing.

**Texts**: 
* $C = \{d_i\}$ corpus of mostly unstructured natural language texts. It is finite but arbitrarily large → Latency is important. Provided ahead of time. It is assumed that we have a static corpus.
* Issue of text length will be important to transformers.

**Queries**: 
* Queries $\neq$ information needs.
* Information needs to be encoded in some representation to enable automated processing.
* Which kind of expression of the information need is fed to the model can have an effect on the retrieval effectiveness? (This happens in BERT)

**Text Ranking Problem**: (aka top k retrieval) 
* Given a query $q$ that expresses a need of information and corpus of documents $$C$$, the text ranking task purpose is to return a ranked list of $k$ text from the corpus collection that maximizes a particular metric of interest.
* The parameter $k$ is the retrieval depth.

**Relevance**: relation between the text and the information needed (in the eyes of an assessor). Being relevant means that the text addresses the information in need. IR works with topical relevance. Relevance depends on the subject the ranking is addressed to.

**Ranking Metrics**: they quantify the quality of a ranking of texts from relevant judgements $(q,d,r)$. The judgement usually comes from an annotation and describes the relevancy r of document d to query q. It can be a binary value, a five points scale... 
* The qrels are the relevance judgements (ground truth) and the run is the predicted scores. These two rankings are fed into a trec_eval: automatically computes metrics.
* **Precision**: fraction of documents in a ranked list $R$ that are relevant. $\text{Precision}(R,q) = \frac{\sum_{(i,d)\in R}\text{rel}(q,d)}{|R|}$, where $\text{rel}(q,d)$ is the binary relevance of document $d$ to query $q$. P@k would be the cutoff precision, it can be understood as of the k top results which fraction are relevant. 
    * **R-Precision**: cutoff precision of relevant documents for a particular topic.
    * Advantage: Easy to interpret.
    * Downside: does not take into account graded relevance, only binary.
    * Downside: does not take into account the ranking positions, you will get the same P@5 if the relevant documents are the number 1 and 2 or if the relevant documents are the number 4 and 5. In both cases P@5 = 0.4, but we as users would prefer the first ranked list.
* **Recall**: fraction of relevant documents for $q$ in the entire corpus $C$ that are retrieved in the ranked list $R$. $\text{Recall}(R,q) = \frac{\sum_{(i,d)\in R}\text{rel}(q,d)}{\sum_{d\in \mathcal{C}}\text{rel}(q,d)}$. It assumes binary relevance. R@k if we evaluate the recall at a cutoff k. 
    * Advantage: Easy to interpret.
    * Downside: does not take into account graded relevance, only binary.
    * Downside: does not take into account the ranking positions
* **Reciprocal Rank (RR)**: $\text{RR}(R,q) = \frac{1}{\text{rank}_i}$, $\text{rank}_i$ is the smallest rank number of a relevant document, is the first relevant document appears at position 1 then the RR is 1, if it appears at position 3 then RR is 1/3, if it appears at position 6 then RR is 1/6... 
    * Only captures the appearance of the first relevant result.
    * Poor choice for ad hoc retrieval. Good for QA.
* **Average Precision (AP)**: $\text{AP}(R,q) = \frac{\sum_{(i,d)\in R}\text{Precision}@i(R,q)\cdot \text{rel}(q,d)}{\sum_{d\in \mathcal{C}}\text{rel}(q,d)}$, it averages the precision scores at different cutoff corresponding to the appearance of the relevant documents. 
    * It captures both precision and recall and favors relevant document appearing at the top of the ranked list.
* **Normalized Discounted Cumulative Gain (nDCG)**: for web search, and it works with non-binary relevance measures. 
    * $\text{DCG}(R,q) = \sum_{(i,d)\in R}\frac{2^{\text{rel}(q,d)}-1}{\log_2(i+1)}$, relevant results near the top are more worth than the others.
    * $\text{nDCG}(R,q) = \frac{\text{DCG}(R,q)}{\text{IDCG}(R,q)}$, $\text{IDCG}$ represents the ideal ranked list. We just normalize with the best possible list so [0,1].
* As a single summary statistic, we can use the arithmetic mean across different topics. MAP is mean precision, MRR is the mean reciprocal rank,
* Judged@k: fraction of judged documents. It is important as unjudged documents are treated as not relevant.
* How do we break ties between documents in a ranked list? In an arbitrary way?
* **Ranked Based Precision (RBP)**.

Test collections have a set of documents of length $|\mathcal{C}|$ and an average length of the documents $\bar{L}(\mathcal{C})$, the number of queries $|q|$ and the average length of those $\bar{L}(q)$, the amount of relevance judgements

Conclusions from only experiments on one test collection should be avoided and instead we should evaluate our model on multiple test collections.

**Keyword search**: techniques that rely on exact term matching to compute the relevant scores between queries and documents from a corpus. Usually with bag-of-words queries but now always, also we can use n-grams.

----
****
