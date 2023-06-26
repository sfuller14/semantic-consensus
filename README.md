# Commercial Consensus

Pinecone/Devpost Hackathon June 2023  
Try it out: [Commercial Consensus](http://ecommerce-recsys.us-east-2.elasticbeanstalk.com)  (hosted on AWS)

## Overview

Commercial Consensus is an intelligent search platform designed to revolutionize e-commerce recommendation systems. By combining the power of structured tabular data with unstructured customer reviews, we provide an innovative solution to the longstanding problem of data quality in e-commerce platforms.

## The Problem

Traditional e-commerce recommendation systems, such as collaborative filtering and graph-based methods, rely heavily on structured, tabular data. However, this approach is fraught with limitations due to the widespread missing and inconsistent data inherent to third-party seller platforms:  

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

Our platform features a chat interface that leverages the power of quality-controlled retrieval-augmented generation (RAG). By utilizing a custom `pinecone.query()` + `cohere.rerank()` + `openai.ChatCompletions()` chain, users receive detailed responses based on both product specifications and relevant reviews. This feature not only provides valuable insights to potential buyers but also serves as a tool for sellers performing market research.

---

<p align="center" style="font-size:10px;">Access aspect-based sentiments from reviews</p>

![Chat Example](https://github.com/sfuller14/semantic-consensus/assets/54780092/ddf82542-d5cf-4d92-ab88-25e50a8831ff)
<p align="center" style="font-size:10px;">Custom pinecone.query() + cohere.rerank() + openai.ChatCompletions() chain.</p>

---

<p align="center" style="font-size:10px;">Targeted retrieval + controlled generation using both aggregated reviews and product specs.</p>

![Screenshot 2023-06-25 at 8 53 58 PM](https://github.com/sfuller14/semantic-consensus/assets/54780092/bf75e849-8d9f-4981-bfcf-5c17616869bf)

---
## Libraries

Commercial Consensus is built using a suite of cutting-edge technologies. Pinecone enables efficient vector similarity search over large volumes of data organized by namespace (product embeddings, word embeddings, and combined) and stored with metadata. Cohere's reranking capabilities are utilized heavily to enhance the relevance of the inital query results and ensure high quality documents are retrieved for GPT-4 in the Chat portion. A customized prompt (as well as providing product specs along with product reviews) limits hallucination and maximizes response relevancy by OpenAI's GPT-4 model.  Streamlit enabled a backend dev to throw a passable front-end on the system.

## Commercial Consensus

The name "Commercial Consensus" embodies the core value proposition of our platform. By aggregating and semantically searching customer reviews, we are able to capture the collective wisdom of the consumer base - a consensus on the quality and features of products. This consensus, driven by real user experiences, provides a reliable and unbiased source of information that complements, and often outperforms, traditional tabular data.

## Try It Out

[Experience the future of e-commerce with Commercial Consensus](http://ecommerce-recsys.us-east-2.elasticbeanstalk.com). Hosted on AWS.

## Narrated Demo:
**NOTE** - video & GIF were filmed before final optimized search process was implemented. The video gives a decent overview of the first draft, but please refer to screenshots or, better yet, try out the app!  

https://youtu.be/5KyWZLdwDzo

## GIF:
**NOTE** - video & GIF were filmed before final optimized search process was implemented. Please refer to screenshots or, better yet, try out the app!  

https://github.com/sfuller14/semantic-consensus/assets/54780092/4d7962a7-9db2-480c-9284-988feb9cdc1a

## Technical Appendix

Re-ranking is an important and widely-used step in modern search engines. It is generally run on the results of a lighter-weight lexical search (like TF-IDF or BM25) to refine the results. Re-ranking using BERT variants has shown SOTA search status in recent years:
- [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/pdf/2004.12832.pdf)

- [Passage Re-ranking with BERT](https://arxiv.org/pdf/1901.04085.pdf)

Cohere recently introduced their [rerank endpoint](https://txt.cohere.com/rerank/) along with some research on the associated search improvement it provides:
<p float="center">
  <img src="https://github.com/sfuller14/semantic-consensus/assets/54780092/04641b8a-5745-4fe5-bd04-d18a8db7f353" width="350" />
</p>

We found the endpoint to be highly performant, both in terms of quality and response time. It handles up to 1,000 documents in a single call _passed as plain text, not embeddings_ and returns the re-ranked results almost instantly.  
#### Each call made to pinecone.query() in ```main.py``` is followed by co.rerank()  
This occurs at three points in our application:
* When the user enters a query and presses Search
     * The top 750 reviews are returned from the metadata-filtered namespace in pinecone
     * These are passed to co.rerank() which returns the top 78x4
     * Duplicate products are removed from the re-ranked results and the top 78 are returned to the 'Search' screen as recommendations
     * re-rank's score * 100 is displayed as similarity in the tooltip with the closest review (though this is probably misleading)
 * When a user clicks View on a product
     * The top 50 reviews are returned using pinecone.query() (which is using cosine sim since the embeddings are text-ada-002)
     * co.rerank is used to get the top 5
 * When a user enters a question in the Chat tab
     * The top 100 reviews are returned using pinecone.query()
     * co.rerank is used to get the top 12
     * The user query, product's title (which for Amazon contains a hodgepodge of specs) + those reviews + the system prompt are passed to GPT-4 (with tiktoken truncating the reviews if cl100k_base max context window is exceeded)
       * This approach (and the system prompt) ensure high quality results of the RAG process and prevent max context window errors
