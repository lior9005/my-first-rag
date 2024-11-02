from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import key_param


dbName = "rag_db"
collectionName = "rag_bd"
index = "vector_index" # atlas vector search name

# Initializing a MongoDB vector search instance
vectorStore = MongoDBAtlasVectorSearch.from_connection_string(
    key_param.MONGODB_URI,
    dbName + "." + collectionName,
    OpenAIEmbeddings(disallowed_special=(), openai_api_key=key_param.LLM_API_KEY), # Embeddings model setup
    index_name=index,
)

def query_data(query):
    # Setting up the retriever to search for similar documents based on the query
    retriever = vectorStore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 3,
            "pre_filter": { "hasCode": { "$eq": False } },
            "score_threshold": 0.01
        },
    )

    # Defining a prompt template to guide the model's response
    template = """
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Do not answer the question if there is no given context.
    Do not answer the question if it is not related to the context.
    Do not give recommendations to anything other than MongoDB.
    Context:
    {context}
    Question: {question}
    """

    # Creating a prompt template object using the defined template
    custom_rag_prompt = PromptTemplate.from_template(template)

    # Setting up the retrieval process - combines documents and formats context for the model
    retrieve = {
        "context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])), 
        "question": RunnablePassthrough() # Passes the original query directly to the model
        }
    
    # Initializing the language model with low temperature for deterministic responses
    llm = ChatOpenAI(openai_api_key=key_param.LLM_API_KEY, temperature=0)

    # Setting up a parser to convert LLM output to a string
    response_parser = StrOutputParser()

    rag_chain = (
        retrieve
        | custom_rag_prompt # Applies the prompt template
        | llm               # Passes context and question to the LLM for generating the response
        | response_parser   # Converts the LLM output to a string
    )

    answer = rag_chain.invoke(query)
    

    return answer

# example query
query_data("When did MongoDB begin supporting multi-document transactions?")
