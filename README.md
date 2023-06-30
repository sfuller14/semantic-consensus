# Commercial Consensus

Pinecone/Devpost Hackathon June 2023  
- Try it out: [Commercial Consensus](http://ecommerce-recsys.us-east-2.elasticbeanstalk.com)  (hosted on AWS)
- [Libraries & Execution flow diagrams](#execution-flow)
- Narrated Demo: https://youtu.be/5KyWZLdwDzo **NOTE** - video is of first draft implementation. See GIF for updated version (or, better yet, try out the app)!
- [Inspiration & references](#inspiration-and-references)
- [DIY locally](#builders)

## Demo

![](https://github.com/sfuller14/public_ref/blob/master/recsys.gif)

## The Problem

Traditional e-commerce recommendation systems, such as collaborative filtering and graph-based methods, rely heavily on structured, tabular data. However, this approach is fraught with limitations due to the widespread missing and inconsistent data inherent to third-party seller platforms:  

## Overview

Commercial Consensus is an intelligent search platform designed to revolutionize e-commerce recommendation systems. By combining the power of structured tabular data with unstructured customer reviews, we provide an innovative solution to the longstanding problem of data quality in e-commerce platforms.

---

<p align="center" style="font-size:10px;">Example of inconsistent data availability for two products in the same category:</p>

<p float="left">
  <img src="https://github.com/sfuller14/semantic-consensus/assets/54780092/43f1c875-05bd-419f-9bbe-1e005dbad521" width="450" />
  <img src="https://github.com/sfuller14/semantic-consensus/assets/54780092/1d0f548d-4f87-409b-bae3-408c51bcc7a1" width="450" /> 
</p>

<p align="center" style="font-size:10px;">Missing data across our full dataset:</p>

![image](https://github.com/sfuller14/semantic-consensus/assets/54780092/fd218f4d-5b0a-4acd-93b0-9893f8c6530f)

<p align="center" style="font-size:10px;">Even when data is available, it is often heterogeneous:</p>

![Screenshot 2023-06-26 at 9 00 26 AM](https://github.com/sfuller14/semantic-consensus/assets/54780092/863814ae-15af-4595-8ccb-220c89d08d65)

---

This data quality issue hampers the effectiveness of recommendation systems, thereby reducing platform revenue generation as well as impeding optimal user experience.

## The Solution

Commercial Consensus tackles this problem head-on by harnessing the latent information within customer reviews. Utilizing state-of-the-art technologies, such as Pinecone's vector search engine over indexed product review embeddings and Cohere's reranking endpoint, our platform performs a hybrid (tabular + semantic) search. This innovative approach provides a new dimension of search, enabling users to tap into the ___Commercial Consensus___ – an aggregated, reliable body of knowledge derived from customer reviews - in a targeted and personalized way.

## Features

### [Enhanced](#technical-appendix) Search

Commercial Consensus offers an ultra-personalized and efficient search experience, powered by `pinecone.query()` and `cohere.rerank()`. Our platform goes beyond lexical search/simple keyword matching, delivering results based on [enhanced](#technical-appendix) semantic similarity to user queries. This approach provides a more intuitive and user-centric search experience, improving product discovery and enhancing user satisfaction.  

---

<p align="center" style="font-size:10px;">Ultra-relevant recommendations using pinecone.query() (with metadata & namespace filters) + cohere.rerank()</p>

![Search Example](https://github.com/sfuller14/semantic-consensus/assets/54780092/27f4c830-c869-4f6b-a859-77fb87b68f6e)
<p align="center" style="font-size:10px;">Hover over the '?' icon to see the most similar review to your query.</p>

---

<p align="center" style="font-size:10px;">'View' page contains detailed product specs and relevant reviews</p>

![2](https://github.com/sfuller14/semantic-consensus/assets/54780092/2b1af8b9-d7a4-47f7-972f-638fd9ae792a)

---
### Intelligent Chat Interface

Our platform features a chat interface that leverages the power of quality-controlled retrieval-augmented generation (RAG). By utilizing a custom `pinecone.query()` + `cohere.rerank()` + `openai.ChatCompletions()` chain, users receive detailed responses based on both product specifications and relevant reviews. Customers find the right product easier & return fewer items -- a win for all parties. This feature not only provides valuable insights to potential buyers but also serves as a tool for sellers performing market research.

---

<p align="center" style="font-size:10px;">Access aspect-based sentiments from reviews</p>

![Chat Example](https://github.com/sfuller14/semantic-consensus/assets/54780092/ddf82542-d5cf-4d92-ab88-25e50a8831ff)
<p align="center" style="font-size:10px;">Custom pinecone.query() + cohere.rerank() + openai.ChatCompletions() chain.</p>

---

<p align="center" style="font-size:10px;">Targeted retrieval + controlled generation using both aggregated reviews and product specs.</p>

![Screenshot 2023-06-25 at 8 53 58 PM](https://github.com/sfuller14/semantic-consensus/assets/54780092/bf75e849-8d9f-4981-bfcf-5c17616869bf)

---
## Libraries

Commercial Consensus is built using a suite of [cutting-edge technologies](#execution-flow). Pinecone enables efficient vector similarity search over large volumes of data organized by namespace (product embeddings, word embeddings, and combined) and stored with metadata. Cohere's reranking capabilities are utilized heavily to enhance the relevance of the inital query results and ensure high quality documents are retrieved for GPT-4 in the Chat portion. A customized prompt (as well as providing product specs along with product reviews) limits hallucination and maximizes response relevancy by OpenAI's GPT-4 model.  Streamlit enabled a backend dev to throw a passable front-end on the system.

## Commercial Consensus

The name "Commercial Consensus" embodies the core value proposition of our platform. By aggregating and semantically searching customer reviews, we are able to capture the collective wisdom of the consumer base - a consensus on the quality and features of products. This consensus, driven by real user experiences, provides a reliable and unbiased source of information that complements, and often outperforms, traditional tabular data.

## Try It Out

[Experience the future of e-commerce with Commercial Consensus](http://ecommerce-recsys.us-east-2.elasticbeanstalk.com). Hosted on AWS.

## Technical Appendix

Re-ranking is an important and widely-used step in modern search engines. It is generally run on the results of a lighter-weight lexical search (like TF-IDF or BM25) to refine the results. Re-ranking using BERT variants has shown SOTA search status in recent years:
- [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/pdf/2004.12832.pdf)

- [Passage Re-ranking with BERT](https://arxiv.org/pdf/1901.04085.pdf)

Cohere recently introduced their [rerank endpoint](https://txt.cohere.com/rerank/) along with some research on the associated search improvement it provides:
<p float="center">
  <img src="https://github.com/sfuller14/semantic-consensus/assets/54780092/04641b8a-5745-4fe5-bd04-d18a8db7f353" width="350" />
</p>

We found the endpoint to be highly performant, both in terms of quality and response time. It handles up to 1,000 documents (passed as raw text, not embeddings) in a single call, and returns the re-ranked results almost instantly.   

__Each call made to pinecone.query() in ```main.py``` is followed by co.rerank(). This occurs at three points in our application__: 

### Execution Flow

1) When the user enters a query and presses 'Search'

![Screenshot 2023-06-28 at 9 36 34 PM](https://github.com/sfuller14/semantic-consensus/assets/54780092/c118d859-6adc-4bbe-8ff8-bc0745ba356b)

__EVEN THOUGH THIS IS POTENTIALLY CONFUSING TO THE USER,__ ```rerank_score * 100``` is displayed as 'Similarity' in the tooltip on hover ([to try to get a sense of how to set threshold](https://docs.cohere.com/docs/reranking-best-practices#interpreting-results) since this is just demo app)

2) When a user clicks View on a product

![Screenshot 2023-06-28 at 9 37 57 PM](https://github.com/sfuller14/semantic-consensus/assets/54780092/1f7f62a4-c590-48ab-a707-666200672a06)
     
3) When a user enters a question in the Chat tab

![Screenshot 2023-06-28 at 9 39 22 PM](https://github.com/sfuller14/semantic-consensus/assets/54780092/70f010d8-26f8-462c-b937-8857d3b3409f)

The user question + product's title (__which for Amazon contains a hodgepodge of specs__) + top 12 reviews + the system prompt are passed to ```openai.ChatCompletion.create()``` (with tiktoken truncating the reviews if cl100k_base max context window is exceeded): 

#### Product Title Example

![Screenshot 2023-06-28 at 8 16 23 PM](https://github.com/sfuller14/semantic-consensus/assets/54780092/a76b56a2-097e-4402-bfa7-7639baef65dd)

This is likely done for facilitating lexical search in the presence of variably-populated data fields. We're able to exploit this approach within the LLM prompt. See [the awesome examples from the pinecone documentation](#inspiration-and-references) for more on this topic in the context of querying. Combining sparse-dense search with our tabular-dense approach (and adding in reranking) could be an interesting area for further investigation. 

---

While ```pinecone.query()``` without re-ranking was often sufficient for simple and well-formed queries, certain query formations (like specific negation expressions) led to undesirable results. Adding re-ranking also generally appeared to show better matching on longer reviews, however in some cases this not necessarily desirable (i.e. re-ranking led to longer reviews being prioritized while a more succinct match would be preferred for display on the home page). __In other cases (specifically during RAG chaining), the longer reviews led to significantly better output.__ More testing is needed here.

__A few examples of using ```pinecone.query()``` alone vs. ```pinecone.query()```+```cohere.rerank()```:__

![Screenshot 2023-06-26 at 9 37 22 PM](https://github.com/sfuller14/semantic-consensus/assets/54780092/3f564654-ff9e-4d95-ae0a-1c187f4d6658)

In the above, notice that both reviews mentioning BSOD in the re-ranked results go on to say that they resolved it. 

![Screenshot 2023-06-26 at 11 08 17 PM](https://github.com/sfuller14/semantic-consensus/assets/54780092/4e209d2a-1749-4312-bd98-f00e757522c0)

Note that these comparisons are not reflective of pinecone's querying performance, but of cosine similarity search on 'text-embedding-ada-002'  vs. the re-ranked equivalent.

---

# Inspiration and References

The materials provided by pinecone's documentation are extremely high quality. 

* [This post by James Briggs](https://docs.pinecone.io/docs/ecommerce-search#hybrid-search-for-e-commerce-with-pinecone) on hybrid (sparse textual + dense image) search was the main inspiration for this project & the proposed system is largely a riff/expansion on that case study (with the main additions being tabular filtering, reranking, and LLM incorporation) 
   * [Accompanying Youtube video](https://www.youtube.com/watch?v=AELtGhiAqio) 
   * [Documentation on sparse-dense search in pinecone](https://docs.pinecone.io/docs/hybrid-search) 
   * [This Youtube series on Streamlit was also very helpful](https://www.youtube.com/watch?v=lYDiSCDcxmc) 
   * [As was this Youtube video on Metadata filtering](https://www.youtube.com/watch?v=tn_Y19oB5bs)  
* Chat feature inspired by LangChain's [ConversationalRetrieverChain](https://python.langchain.com/docs/modules/chains/popular/chat_vector_db) with reranking added

--- 

# Builders

See the streamlit-deploy branch
