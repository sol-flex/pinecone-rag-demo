import { Pinecone } from '@pinecone-database/pinecone';
import { PineconeStore } from '@langchain/pinecone';
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import dotenv from 'dotenv';
import { OpenAIEmbeddings } from "@langchain/openai";
import { pull } from "langchain/hub";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatOpenAI } from "@langchain/openai";


// Load environment variables from .env file
dotenv.config();


const loader = new PDFLoader("Learning_React.pdf");

const docs = await loader.load();

console.log(docs);

//split text 

const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });

const splits = await textSplitter.splitDocuments(docs);

console.log(splits.length)

const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });

const pineconeIndex = pc.Index("demo-index");

/*
await pc.createIndex({
  name: indexName,
  dimension: 1536,
  metric: 'cosine',
  spec: { 
    serverless: { 
      cloud: 'aws', 
      region: 'us-east-1' 
    }
  } 
}); 
*/

//create embeddings

const embeddings = new OpenAIEmbeddings({
    model: "text-embedding-3-small",
    apiKey: process.env.OPENAI_API_KEY
});

// uploading vectors to pinecone for the first time

// const docsearch = await PineconeStore.fromDocuments(docs, embeddings, { pineconeIndex })

const docsearch = await PineconeStore.fromExistingIndex(embeddings, {
    pineconeIndex,
    // Maximum number of batch requests to allow at once. Each batch is 1000 vectors.
    maxConcurrency: 5,
    // You can pass a namespace here too
    // namespace: "foo",
});
  

console.log(docsearch)

const stats = await pineconeIndex.describeIndexStats();

console.log(stats)

const retriever = docsearch.asRetriever()
const prompt = await pull("rlm/rag-prompt");

const llm = new ChatOpenAI({ model: "gpt-3.5-turbo", temperature: 0, apiKey: process.env.OPENAI_API_KEY });



const ragChain = await createStuffDocumentsChain({
    llm,
    prompt,
    outputParser: new StringOutputParser(),
  });
  
  const retrievedDocs = await retriever.invoke("what is useEffect?");
  
  console.log(retrievedDocs)

  const res = await ragChain.invoke({
    question: "what is useEffect?",
    context: retrievedDocs,
  });
  
  console.log(res);