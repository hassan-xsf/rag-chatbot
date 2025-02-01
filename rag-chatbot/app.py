from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.schema import SystemMessage , HumanMessage , AIMessage
import streamlit as st

st.title("🎓 SMIU AI Agent")
st.caption("🚀 An advanced SMIU agent made using Gemini + Langchain RAG")

load_dotenv()

@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def load_faiss_vector(embeddings):
    return FAISS.load_local(
        "./smiu-db", 
        embeddings, 
        allow_dangerous_deserialization=True
    )

@st.cache_resource(show_spinner=False)
def load_llm():
    return ChatGoogleGenerativeAI(model="gemini-1.5-pro")

embeddings = load_embeddings()
faiss_vector = load_faiss_vector(embeddings)
llm = load_llm()


systemPrompt = "You are an AI assistant for SMIU that answers questions based on the data provided. Your goal is to provide accurate and relevant information to the user based on the context provider\
        Context: {context}\
        Query: {query}"
    
prompt = PromptTemplate(
    template=systemPrompt,
    input_variables=["context", "query"],
)
chain = prompt | llm



if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you today?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_input = st.chat_input("Enter your queries")



def callLLM(query: str):
    search_results = faiss_vector.similarity_search(query, k=3)
    retrieved_context = "\n\n".join([result.page_content for result in search_results])


    messages = [SystemMessage(content = systemPrompt)]
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            messages.append(HumanMessage(content = msg["content"]))
        else:
            messages.append(AIMessage(content = msg["content"]))

    response = chain.invoke({
        "context": retrieved_context, 
        "query": query,
        "messages": messages
    }).content

    return response


if user_input:
    msg = user_input

    st.session_state["messages"].append({"role" : "user", "content" : msg})
    st.chat_message("user").write(msg)

    with st.spinner("Thinking..."):
        response = callLLM(msg)

    st.chat_message("assistant").write(response)
    st.session_state["messages"].append({"role" : "assistant", "content" : response})
