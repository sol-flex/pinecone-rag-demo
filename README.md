###This script uses Langchain, Pinecone, and OpenAI to create a RAG application over a PDF.###
/n
1. The PDF is converted into vectors /n
2. The vectors are uploaded to Pinecone /n
3. The vectorstore is used as a retriever /n
4. A RAG chain is created to query the vector store and answer question /n
/n
npm init
npm install 
node index.js