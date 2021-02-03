---
layout: post
title: Paper Summary of "Pretrained Transformer for Text Ranking - Bert and Beyond"
---

### Introduction
**Text Ranking**: generate an ordered set of texts retrieved from a corpus in response to a query for a particular task. The most common form of text ranking is search.
    - Examples: QA Question Answering, CQA Community QA, Information Filtering, Text Recommendation, Ranking as Input to Downstream Models.
    - Examples beyond information access: Distant Supervision and Data Augmentation, Selecting from Competing Hypotheses.

**Ad Hoc Retrieval Problem**: we want to retrieve the relevant texts for a query that are about the topic of the user's request and address the need of information of the user.

**Exact Term Matching**: text from docs and texts from queries have to match exactly to contribute to the ranking.
    - It's a challenge when the query describes what it wants to search in different words as the documents. Then there is no exact match.

Deep Learning has freed the text ranking problem from the challenges of exact term matching.


The easiest way to make your first post is to edit this one. Go into /_posts/ and update the Hello World markdown file. For more instructions head over to the [Jekyll Now repository](https://github.com/barryclark/jekyll-now) on GitHub.

----
****
