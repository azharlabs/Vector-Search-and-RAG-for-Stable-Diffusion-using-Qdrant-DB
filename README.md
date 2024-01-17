# Vector Search and RAG for Stable Diffusion using Qdrant DB

To assist users in this task, a sophisticated system using Vector Search and Retrieval Augmented Generation (RAG) can be employed. This system aims to analyze a vast database of successful prompts, identifying and suggesting the most relevant ones to the user's input, thus streamlining the process of initiating Stable Diffusion.

### Install the required python libraries

Before proceeding, ensure that all necessary Python libraries are installed. These libraries are essential for running the subsequent code.

Here is the list of libraries:
 - datasets==2.13.0
 - diffusers==0.25.0
 - FlagEmbedding==1.1.8
 - huggingface-hub==0.20.2
 - langchain==0.1.1
 - langchain-community==0.0.13
 - Levenshtein==0.23.0
 - openai==1.8.0
 - pandas==2.1.4
 - pypdf==3.17.4
 - qdrant-client==1.7.0
 - requests==2.31.0
 - sentence-transformers==2.2.2
 - SQLAlchemy==2.0.25
 - streamlit==1.26.0
 - torch==2.1.2
 - torchvision==0.16.2
 - transformers==4.36.2

Install them using the following command:

```
pip install -r requirements.txt
```
### Add the huggingface access token in img_rag_lib.py
```
hf_token = ""
```
### Prepare the dataset of 1K prompt examples from DiffusionDB

This step involves downloading and preparing a dataset of 1,000 prompt examples from HuggingFace's DiffusionDB. After execution, it generates a CSV file named prompts_unique.csv.

```
python imgrag_prep.py
```
### Run the application

With the data prepared, you're ready to build and run the application. This process converts the dataset/prompts into embeddings and builds an in-memory vector store in Qdrant for semantic search.

Run the application using the following command:

 ```
streamlit run img_app.py --server.port 8080
```
