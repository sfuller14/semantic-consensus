# Commercial Consensus: Revolutionizing E-Commerce with Intelligent Semantic Search

Pinecone/Devpost Hackathon June 2023  
Try it out: [Commercial Consensus](http://ecommerce-recsys.us-east-2.elasticbeanstalk.com)  (hosted on AWS)

## Overview

Commercial Consensus is an intelligent search platform designed to revolutionize e-commerce recommendation systems. By combining the power of structured tabular data with unstructured customer reviews, we provide an innovative solution to the longstanding problem of data quality in e-commerce platforms.

## The Problem

Traditional e-commerce recommendation systems, such as collaborative filtering and graph-based methods, rely heavily on structured, tabular data[^1]. However, this approach is fraught with limitations due to the widespread missing and inconsistent data inherent to third-party seller platforms:  

---

<p align="center" style="font-size:10px;">Example of inconsistent data availability for two products in the same category:</p>

<p float="left">
  <img src="https://github.com/sfuller14/semantic-consensus/assets/54780092/43f1c875-05bd-419f-9bbe-1e005dbad521" width="400" />
  <img src="https://github.com/sfuller14/semantic-consensus/assets/54780092/1d0f548d-4f87-409b-bae3-408c51bcc7a1" width="400" /> 
</p>

<p align="center" style="font-size:10px;">Missing data across our full dataset:</p>

![heatmap](https://github.com/sfuller14/semantic-consensus/assets/54780092/031494eb-d718-4f91-9a4a-860d975a15c9)

---

This data quality issue hampers the effectiveness of recommendation systems, thereby reducing platform revenue generation as well as impeding optimal user experience.

## The Solution

Commercial Consensus tackles this problem head-on by harnessing the latent information within customer reviews. Utilizing state-of-the-art technologies, such as Pinecone's vector search engine over indexed product review emebddings and Cohere's reranking endpoint, our platform performs a hybrid (tabular + semantic) search. This innovative approach provides a new dimension of search, enabling users to tap into the ___Commercial Consensus___ – an aggregated, reliable body of knowledge derived from customer reviews - in a targeted and personalized way.

## Features

### Enhanced Search

Commercial Consensus offers an ultra-relevant search experience, powered by `pinecone.query()` and `cohere.rerank()`. Our platform goes beyond simple keyword matching, delivering results based on semantic similarity to user queries. This approach provides a more intuitive and user-centric search experience, improving product discovery and enhancing user satisfaction.  

---
![Search Example](https://github.com/sfuller14/semantic-consensus/assets/54780092/27f4c830-c869-4f6b-a859-77fb87b68f6e)
<p align="center" style="font-size:10px;">Ultra-relevant recommendations using pinecone.query() (with metadata & namespace filters) + cohere.rerank(). <br> Hover over the '?' icon to see the most similar review to your query.</p>

---

![2](https://github.com/sfuller14/semantic-consensus/assets/54780092/2b1af8b9-d7a4-47f7-972f-638fd9ae792a)
<p align="center" style="font-size:10px;">'View' page contains detailed product specs and relevant reviews</p>

---
### Intelligent Chat Interface

Our platform features a chat interface that leverages the power of retrieval-augmented generation (RAG). By utilizing a custom `pinecone.query()` + `cohere.rerank()` + `openai.ChatCompletions()` chain, users can ask questions and receive detailed responses based on both product specifications and relevant reviews. This feature not only provides valuable insights to users but also serves as a tool for sellers performing market research.

---
![Chat Example](https://github.com/sfuller14/semantic-consensus/assets/54780092/ddf82542-d5cf-4d92-ab88-25e50a8831ff)
<p align="center" style="font-size:10px;">Access aspect-based sentiments from reviews. <br> Custom pinecone.query() + cohere.rerank() + openai.ChatCompletions() chain.</p>

---
![Screenshot 2023-06-25 at 8 53 58 PM](https://github.com/sfuller14/semantic-consensus/assets/54780092/bf75e849-8d9f-4981-bfcf-5c17616869bf)
<p align="center" style="font-size:10px;">Targeted retrieval + controlled generation using both aggregated reviews and product specs.</p>

---
## Libraries

Commercial Consensus is built using a suite of cutting-edge technologies. Pinecone, a vector database, enables efficient vector similarity search over large volumes of data. Cohere's reranking capabilities are utilized heavily to enhance the relevance of search results, while OpenAI's language model provides sophisticated natural language generation capabilities. This combination of technologies transforms the way we access and interpret e-commerce data.

## Commercial Consensus

The name "Commercial Consensus" embodies the core value proposition of our platform. By aggregating and semantically searching customer reviews, we are able to capture the collective wisdom of the consumer base - a consensus on the quality and features of products. This consensus, driven by real user experiences, provides a reliable and unbiased source of information that complements, and often outperforms, traditional tabular data.

## Try It Out

Experience the future of e-commerce with Commercial Consensus. [Discover a new, intelligent way to navigate online shopping platforms.](http://ecommerce-recsys.us-east-2.elasticbeanstalk.com). Hosted on AWS.

[^1]: Source + validation of project: <img width="340" alt="Screenshot 2023-06-20 at 8 18 45 AM" src="https://github.com/sfuller14/semantic-consensus/assets/54780092/01e6ebdf-1dbb-41ad-b9dd-09829ad495dc">


