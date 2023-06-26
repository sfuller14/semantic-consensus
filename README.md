# E-Commerce Intelligent Search Platform
Pinecone/Devpost Hackathon June 2023  
Try it out: http://ecommerce-recsys.us-east-2.elasticbeanstalk.com

## Overview
Intelligent search platform/recsys for the top 1,000 Amazon desktops & laptops  
Chat with reviews to get unbiased product information

## Search
##### Natural language querying over reviews --> Unbiased user-tailored recommendations
Ultra-relevant search results using ```pinecone.query()``` (with metadata & namespace filters) + ```cohere.rerank()```  

<p align="center">Hover over the ? icon to see the most similar review to your query:</p>  

![1](https://github.com/sfuller14/semantic-consensus/assets/54780092/27f4c830-c869-4f6b-a859-77fb87b68f6e)

<p align="center">'View' page contains detailed product specs and relevant reviews:</p>  

![2](https://github.com/sfuller14/semantic-consensus/assets/54780092/2b1af8b9-d7a4-47f7-972f-638fd9ae792a)


## Chat
##### Ask questions and get feature-specific sentiments by chatting with reviews and product specs  
Optimized RAG process uses a custom ```pinecone.query()``` + ```cohere.rerank()``` + ```openai.ChatCompletions()``` chain.  
Useful for both users and sellers performing market research.  

<p align="center">Access aspect-based sentiments from reviews:</p>  

![Screenshot 2023-06-25 at 9 28 02 PM](https://github.com/sfuller14/semantic-consensus/assets/54780092/ddf82542-d5cf-4d92-ab88-25e50a8831ff)


<p align="center">Get answers based both on reviews and product specs:</p>  

![Screenshot 2023-06-25 at 8 53 58 PM](https://github.com/sfuller14/semantic-consensus/assets/54780092/bf75e849-8d9f-4981-bfcf-5c17616869bf)


## Libraries
Pinecone + Cohere + OpenAI API + AWS + Streamlit + Apify 

## Narrated Demo:
NOTE -- this video and the below GIF were filmed before incorporating reranking and other features. Please see screenshots or, better yet, try out the updated app!
https://youtu.be/5KyWZLdwDzo

## GIF:
https://github.com/sfuller14/semantic-consensus/assets/54780092/4d7962a7-9db2-480c-9284-988feb9cdc1a
