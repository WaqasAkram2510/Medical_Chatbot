from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec 
from langchain_pinecone import PineconeVectorStore
from src.helper import load_pdf_files, filter_to_minimal_docs, text_split, download_embeddings

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_KEY")

os.environ["PINECOUNE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

extracted_pdf = load_pdf_files(data = 'data/')
print("extraction completed ...........................")
minimal_docs = filter_to_minimal_docs(extracted_pdf)
text_chunk = text_split(minimal_docs)
print("chunking done ...........................")


embedding = download_embeddings()
print("embeding downloaded ...........................")


pinecone_api_key = PINECONE_API_KEY

pc = Pinecone(api_key = pinecone_api_key)

index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name = index_name,
        dimension=384,  # Dimension of the embeddings
        metric= "cosine",  # Cosine similarity
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
print("index created ...........................")

index = pc.Index(index_name)

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunk,
    embedding=embedding,
    index_name=index_name
)
print("Completed ...........................")