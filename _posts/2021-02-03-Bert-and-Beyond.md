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

**Pre-BERT Models**: 
* Representation-based: learn dense vector representation of queries and documents independently, and we compute the similarity between the representation with cosine similarity or inner product. DSSM, 2013. DESM, 2016.
* Interaction-based: compare representation of term of the query and documents directly and produce a similarity matrix. DRMM, 2016. KNRM, 2017.
* Both approaches can include neural networks for creating the representations.
* IB are more effective models but slower than RB.
* Hybrid Models: DUET, 2017.

**BERT** (Bidirectional Encoder Representations from Transformers, Devlin et al. 2019) is an algorithm that uses transformers (like LSTM but better). 
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

* **Precision**: fraction of documents in a ranked list $R$ that are relevant. $\text{Precision}(R,q) = \frac{\sum_{(i,d)\in R}\text{rel}(q,d)}{\lvert R \rvert}$, where $\text{rel}(q,d)$ is the binary relevance of document $d$ to query $q$. $P@k$ would be the cutoff precision, it can be understood as of the k top results which fraction are relevant. 
    * **R-Precision**: cutoff precision of relevant documents for a particular topic.
    * Advantage: Easy to interpret.
    * Downside: does not take into account graded relevance, only binary.
    * Downside: does not take into account the ranking positions, you will get the same P@5 if the relevant documents are the number 1 and 2 or if the relevant documents are the number 4 and 5. In both cases P@5 = 0.4, but we as users would prefer the first ranked list.

* **Recall**: fraction of relevant documents for $q$ in the entire corpus $C$ that are retrieved in the ranked list $R$. $\text{Recall}(R,q) = \frac{\sum_{(i,d)\in R}\text{rel}(q,d)}{\sum_{d\in \mathcal{C}}\text{rel}(q,d)}$. It assumes binary relevance. $R@k$ if we evaluate the recall at a cutoff $k$. 
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

Test collections have a set of documents of length $\lvert\mathcal{C}\rvert$ and an average length of the documents $\bar{L}(\mathcal{C})$, the number of queries $\lvert q\rvert$ and the average length of those $\bar{L}(q)$, the amount of relevance judgements

Conclusions from only experiments on one test collection should be avoided and instead we should evaluate our model on multiple test collections.

**Keyword search**: techniques that rely on exact term matching to compute the relevant scores between queries and documents from a corpus. Usually with bag-of-words queries but now always, also we can use n-grams.

### Multi-Stage Ranking Architectures

**Relevance Classification**: treat the problem as a classification where we need to infer if the documents are in the relevant class and the ranking is just a list of all the order documents by this probability of being the relevant class for an information need. → Supervised Machine Learning. Point-wise learning technique. 
 * It is a simplification as relevance is not a binary property.

Probability Ranking Principle: documents should be ranked in decreasing order of the estimated probability of being relevant with respect to the information needs.

**BERT**: Bidirectional Encoder Representation from Transformers [Devlin et al. 2019]. 
* Neural network model that generates contextual embeddings for input sequences in English.
* Input: vector representation derived from tokens. The input sequences are usually tokenised by WordPiece or BPE. Their aim is to reduce the vocabulary space by splitting words (we can model large texts by using a small vocabulary: 30000 word pieces). The input is divided into three components: 
    * token embedding: that is done by Word Piece.
    * segment embedding: it states if the token belongs to input A or input B ($E_A$ and $E_B$).
    * position embedding: captures the position of the token in the sequence.
    * The final input is an element-wise summation (not a concatenation) of all embeddings.
* The input is passed through a stack of transformer encoder layers that produce the outputs. 
* Hyper-parameters: number of hidden layers, dimension of the hidden layers, attention heads.
* Transformers exhibit quadratic complexity in time and space w.r.t the input.
* Output: sequence of contextual embeddings → context-dependent representations of the input tokens. They capture the complexities of the language like semantics or syntax as well as polysemy 😲.
* Transformers were designed for sequence-to-sequence tasks (video captioning or machine translation) but BERT is just the encoder half of the transformer. (GPT is the decoder half so is the opposite of BERT).
* Uses self-supervision in pre-training compared of starting with random initialization of the model weights. This is good as the model optimization is not bound to the labeled data you have (the text provides its own labels). And it provides a good starting point for more specific tasks.
* The objective is the MLM (Masked Language Model) pre-training objective (Taylor, 1953): we mask a token from the input sequence, and we ask the model to predict it, while training it with a loss.
* It is called Bidirectional as it understands the right and left context of a token in order to make predictions (GPT only understands preceding tokens).
* Tasks: 
    * Single-input classification tasks (sentiment analysis).
    * Two-input classification tasks (detecting if two sentences are paraphrases of each other).
    * Single-input labelling task (named-entity recognition)
    * Two-input labelling tasks (QA, we label the answer span in a text given question)

**monoBERT**: Relevance Classification. We want to estimate the score $s_i$ that represents how relevant is a candidate document $d_i$ to a query $q$: $P(\text{Relevant} = 1\lvert d_i, q)$. 
* Input: [[CLS], q, [SEP], d,[SEP]], q tokens are the verbatim from the user queries.
* Output: a contextual vector representation for each token. The vector $T_{[CLS]}$ of the [CLS] token is imputed to a single layer fully-connected network to obtain the score that represents the relevancy of document d to query q. $P(\text{Relevant} = \lvert 1d_i, q) = s_i = \text{softMax}(T_{[CLS]}W + b)_1$. The factor one is because the single layer has two neurons one for the relevant class and another for the non-relevant class.
* The structure is the BERT and the classification layer. It is trained end-to-end with cross entropy loss.
* Limited to ranking documents of a small length. Why? Because of the limitations of computation nowadays and the fact that BERT was pre-trained with a sequence of tokens smaller than 512 (the positional token is $<$ 512).

**Length limitation of BERT**: tackled by Birch, BERT-MaxP and CEDR (2019). PCGM, PARADE (2020) 
* Training: what do we feed the model? If the query+document exceeds the length of BERT we need to truncate the document so the training will be noisy...
* Inference: what do we feed the model? If we feed it truncated documents how do we compute the overall score of the document? We can do an aggregation of scores or representations of the segments (for example we can take the maximum score).
* **Birch**: avoids the training problem by using data that does not exceed the length and tackles the inference problem by estimating the relevance of sentences and doing aggregation. 
    * They combine the scores of the different sentences by picking the top n scores and combining them with the original document score (the one in the first stage). $s_i = \alpha s_d + (1-\alpha)\sum_{i=1}^nw_is_i$. $\alpha$ and $w_i$ are tuned by cross-validation
    * We can pre-train BERT in a different domain and the results will be applicable to our task with high accuracy.
    * It appears that the document relevance can be estimated considering only the top scoring sentences.
* **Passage Score Aggregation: BERT-MaxP**: it segments documents into overlapping passages. They use a 150-sliding window of stride 75. 
    * Aggregation: $s_d = \max{s_i}$ or $s_d = s_1$ or $s_d = \sum_i s_i$. The max approach is the one that works best
    * The input is not just the title of the topic but a combination of the title and the description.
    * BERT can exploit linguistically rich queries, which is different from keyword search.
    * Extension PCGM (2020).
* **Leveraging Contextual Embeddings: CEDR**: Can we use the contextual embedding of the words to do the ranking? 
    * The training problem is solved by splitting the documents into chunks and BERT inference is applied to every chunk independently. It does an average pooling of the [CLS] representation of the different chunks.
    * The model constructs a similarity matrix as the pre-BERT interaction-based models.
    * The score is a combination of the BERT score and the scores derived from the similarity matrix that go through a fully connected layer.
    * Allows uniform treatment of training and inference.
* **Passage Aggregation Representation: PARADE**: direct descendant from CEDR. 
    * It focusses on aggregation of REPRESENTATION of passages instead of aggregating scores of passages. Aggregates the [CLS] representation of each passage. Passage representations are very rich in information.
    * Differentiable model, it can consider at unison multiple passages.

**Multi-Stage Re-rankers**: 
* Reranking pipelines or cascades or "telescoping": 
    * we have $N$ stages denoted as $H_1$ to $H_N$.
    * The first stage of the ranking is referred as $H_0$, that retrieves $k_0$ from an inverted index.
    * Each stage $H_n$ receives a ranked list $R_{n-1}$ of $k_{n-1}$ candidates, and provides a ranked list $R_n$ of $k_n$ elements.
    * $k_n \leq k_{n-1}$
* A common design is that scores of each stage are additive or a re-ranker can decide to completely ignore the previous scores.
* 💡 Motivation: they balance the tradeoff between effectiveness and efficiency.
* The idea of multi-stage re-rankers is to exploit expensive features only when necessary → early stages uses cheap features to discard easy candidates. They can exploit "early exits" [Cambazoglu et al. 2010].
* For higher recall maybe they are not that useful.
* **Additive ensembles**: the score of each stage is added to the score of the previous stages.
* Pairwise reranking: duoBERT: it focuses on comparing pairs of candidate documents. 
    * $P_{ij} = P(d_i > d_j\lvert d_i,d_j,q)$, which is more relevant $d_i$ or $d_j$, with respect to $q$.
    * The model outputs a comparison between docs, we still need to aggregate this results to form a ranked list.
    * Input: [[CLS],q,[SEP],di,[SEP],dj,[SEP]].
    * For $k$ candidates the output is $\lvert k\rvert \times(\lvert k\rvert-1)$ probabilities.
    * The methods of aggregation can be MAX, MIN, SUM, BINARY.
* **Efficient Multi-Stage Re-rankers: Cascade Transformers**: we want a system even faster but not that accurate. 
    * Early exits in the middle layers of BERT. How do we discard candidates? we can do the bottom 30%, or we can do a score threshold, or learn a classifier.
    * Works only for question answering, not for long documents.

**Document Preprocessing Techniques**: 
* **Vocabulary mismatch problem**: when you use different works to describe a concept. Problem for exact matching techniques. One poor solution could be picking more candidates and that way you will pick all relevant texts.
* The initial candidate generation stage (first stage) it is still a bottle neck.
* **Document expansion**: add additional terms that represent the content of a document. $\neq$ query expansion.
* Document expansion via Query Predictions: **doc2query**: sequence to sequence model that produces queries for a given document. Then the queries are added at the documents like an expansion. It is a very fast technique. → the RECALL is higher! Good for a starting point in downstream models.
* **Term Re-weighting as Regression: DeepCT**: (Deep Contextualized Term Weighting) uses a BERT-based model to output an importance score for each term in a document. 
    * QTR Query Term Recall: $\text{QTR}(t,d) = \frac{\lvert Q_{d,t}\rvert}{\lvert Q_d\rvert} = y_{t,d}$, the denominator is the number of queries relevant to document d and the numerator is the number of queries relevant to document d that contain the term t.
    * $\hat{y}_{t,d} = wT_{t,d} + b$, minimizing the MSE loss.
    * The recall is high! Much faster than doc2query.
* **Term Re-weighting with Weak Supervision: HDCT**: context aware hierarchical document term weighting framework. Same as DeepCT they want to estimate the importance of a term in a document based on the contextual embeddings of BERT. 
    * Aimed at solving the problem of length limitation.
    * They compute the importance scores for terms by passages. We have a vector of all term frequencies in a passage, then a set of this term frequencies at all passages. They do a weighted sum of this vectors.
* **Target Corpus Pre-training and Relevance Transfer**: techniques to use after the first stage and before the rerankers. 
    * TCP: additional pre-training the model with your corpus using the same objective.
    * Out-of-domain Relevance Judgements: provide the model general notions of relevance matching before using task specific data.

BERT is good but slow! What if we go **beyond BERT**? 
* Better pre-trained BERT variants: RoBERTa (removes the next sentence prediction part of the objective), ALBERT (uses the same weights for each layer), ELECTRA (instead of MLM it trains on replaces token detection).
* Distillation: smallest versions of BERT. Student-teacher model. TinyBERT. It degrades effectiveness but increases efficiency. Train a large model and then distill it.
* Re-ranking with Transformers: Transformer Kernel TK(separate transformers stacks to compute contextual representations of query and document terms, the fed to a similarity matrix), TKL (replaces the self-attention layers with local self-attention, attention from a distant term is always 0), Conformer Kernel CK (adds exact term matching component, very memory efficient, low effectiveness)
* Sequence-to-sequence models: monoT5: we can formulate every task as a sequence-to-sequence task. The model outputs true or false depending on the relevance of the document to a query. Very good results but no one understands why.

Domain-specific applications: SciBERT and BioBERT.


----
****
