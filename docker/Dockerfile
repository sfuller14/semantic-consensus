# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container to /app
WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Add the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define environment variable
# ENV NAME World

# # Run app.py when the container launches
# CMD streamlit run main.py --server.enableCORS false --server.address 0.0.0.0 --server.port 8501 --server.enableXsrfProtection false

# AWS Tokens
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG PINECONE_API_KEY
ARG PINECONE_ENVIRONMENT
ARG OPENAI_API_KEY
ARG AWS_RDS_HOSTNAME
ARG AWS_RDS_PORT
ARG AWS_RDS_DB
ARG AWS_RDS_UN
ARG AWS_RDS_PW
ARG COHERE_KEY
ARG AWS_DEFAULT_REGION

ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
ENV PINECONE_API_KEY=$PINECONE_API_KEY
ENV PINECONE_ENVIRONMENT=$PINECONE_ENVIRONMENT
ENV OPENAI_API_KEY=$OPENAI_API_KEY
ENV AWS_RDS_HOSTNAME=$AWS_RDS_HOSTNAME
ENV AWS_RDS_PORT=$AWS_RDS_PORT
ENV AWS_RDS_DB=$AWS_RDS_DB
ENV AWS_RDS_UN=$AWS_RDS_UN
ENV AWS_RDS_PW=$AWS_RDS_PW
ENV COHERE_KEY=$COHERE_KEY
ENV AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]