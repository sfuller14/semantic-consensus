# E-Commerce Intelligent Search Platform
Pinecone/Devpost Hackathon June 2023  
Try it out: http://ecommerce-recsys.us-east-2.elasticbeanstalk.com

## Overview
Hybrid (tabular + semantic) search platform/recommendation system + chatbot for the top 1,000 Amazon desktops & laptops  
* Natural language querying --> User-tailored recommendations
    * __Ultra-relevant, personalized search results using pinecone.query() + cohere.rerank()__
* Ask questions about product specs and get feature-specific sentiments by chatting with reviews
    * __Optimized RAG process by chaining pinecone.query() + cohere.rerank() + openai.ChatCompletions__

## Libraries
Pinecone + Cohere + OpenAI API + AWS + Streamlit + Apify

## Narrated Demo:
https://youtu.be/5KyWZLdwDzo

## GIF:
https://github.com/sfuller14/semantic-consensus/assets/54780092/4d7962a7-9db2-480c-9284-988feb9cdc1a
