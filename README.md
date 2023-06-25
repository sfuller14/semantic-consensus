# E-Commerce Intelligent Search Platform
Pinecone/Devpost Hackathon June 2023  
Try it out: http://ecommerce-recsys.us-east-2.elasticbeanstalk.com

## Overview
Hybrid (tabular + semantic) search platform/recsys for the top 1,000 Amazon desktops & laptops  
Chat with reviews to get unbiased product information

### Search
##### Natural language querying over reviews --> Unbiased user-tailored recommendations
Ultra-relevant search results using ```pinecone.query()``` (with metadata & namespace filters) + ```cohere.rerank()```  

### Chat
##### Ask questions about product specs and get feature-specific sentiments by chatting with reviews
Optimized RAG process uses a custom ```pinecone.query()``` + ```cohere.rerank()``` + ```openai.ChatCompletions()``` chain.  
Useful for both users and sellers performing market research.  

## Libraries
Pinecone + Cohere + OpenAI API + AWS + Streamlit + Apify

## Narrated Demo:
https://youtu.be/5KyWZLdwDzo

## GIF:
https://github.com/sfuller14/semantic-consensus/assets/54780092/4d7962a7-9db2-480c-9284-988feb9cdc1a
