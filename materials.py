from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_transformers.openai_functions import (
    create_metadata_tagger,
)
import key_param

# Connect to MongoDB using URI stored in key_param file
# The database name is "book_mongodb_chunks" and collection name is "chunked_data"
client = MongoClient(key_param.MONGODB_URI)
dbName = "book_mongodb_chunks"
collectionName = "chunked_data"
collection = client[dbName][collectionName]

# Load the PDF file and extract its pages for processing
loader = PyPDFLoader(".\\mongodb.pdf")
pages = loader.load()

# Filter out pages with fewer than 20 words to get only meaningful content
cleaned_pages = []
for page in pages:  
    if len(page.page_content.split(" ")) > 20:
        cleaned_pages.append(page)

# Initialize the text splitter with chunk size and overlap settings
# Chunks each document into segments of 500 characters with an overlap of 150 characters
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)

# Define metadata schema to extract specific metadata fields from each page
schema = {
    "properties": {
        "title": {"type": "string"},       # Title of the document
        "keywords": {"type": "array", "items": {"type": "string"}},  # Keywords as list of strings
        "hasCode": {"type": "boolean"},    # Boolean indicating if code is present
    },
    "required": ["title", "keywords", "hasCode"],  # Required fields in schema
}

# Initialize the language model with OpenAI key for use in metadata tagging
llm = ChatOpenAI(
    openai_api_key=key_param.LLM_API_KEY, temperature=0, model="gpt-3.5-turbo"
)

# Create a document transformer for tagging metadata in documents according to the defined schema
document_transformer = create_metadata_tagger(metadata_schema=schema, llm=llm)

# Transform the documents by tagging them with metadata using the llm model
docs = document_transformer.transform_documents(cleaned_pages)

# Split the transformed documents into smaller chunks for embedding storage
split_docs = text_splitter.split_documents(docs)

# Initialize OpenAI embeddings for each chunk of text to prepare for vector search
embeddings = OpenAIEmbeddings(openai_api_key=key_param.LLM_API_KEY)

# Store the document embeddings in MongoDB Atlas for vector-based searching
vectorStore = MongoDBAtlasVectorSearch.from_documents(
    split_docs, embeddings, collection=collection
)
