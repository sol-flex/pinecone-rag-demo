## RAG Application

**This script uses Langchain, Pinecone, and OpenAI to create a RAG application over a PDF.**

1. The PDF is converted into vectors 
2. The vectors are uploaded to Pinecone 
3. The vectorstore is used as a retriever 
4. A RAG chain is created to query the vector store and answer question 

npm init

npm install 

node index.js