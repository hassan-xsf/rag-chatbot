from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

if __name__ == "__main__":
    faiss_vector = FAISS.load_local(
        "./smiu-db", 
        embeddings, 
        allow_dangerous_deserialization=True
    )
        
    query = "When was SMIU founded?"
    search_results = faiss_vector.similarity_search(query, k=2)


    print(f"\n🔍 Top 2 most relevant chunks for the query: '{query}'\n")
    for i, result in enumerate(search_results, 1):
        print(f"Result {i}:")
        print(f"Content: {result.page_content}\n")

    retrieved_context = "\n\n".join([result.page_content for result in search_results])


    systemPrompt = "You are an AI assistant for SMIU that answers questions based on the data provided. Your goal is to provide accurate and relevant information to the user based on the context provider\
        Context: {context}\
        Query: {query}"
    
    prompt = PromptTemplate(
    template=systemPrompt,
    input_variables=["context", "query"],
    )
    chain = prompt | llm 

    response = chain.invoke({"context": retrieved_context, "query": query}).content

    print(response)


