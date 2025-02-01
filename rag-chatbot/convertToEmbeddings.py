from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

collection_name = "smiu-data"

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def load_data():
    try:
        current_dir = Path.cwd()
        data_path = current_dir / "data" / "smiu-data-original.md"
        with open(data_path, "r", encoding="utf-8") as file:
            text_content = file.read()
        return text_content
    except FileNotFoundError:
        print("Error: Data file not found")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

if __name__ == "__main__":
    text = load_data()
    if text:
        print("Data loaded successfully.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            length_function=len
        )
        splits = splitter.split_text(text)
        print(f"Split the file into {len(splits)} chunks.")

        faiss_vector = FAISS.from_texts(splits, embeddings)
        faiss_vector.save_local("./smiu-db")

        
        print(f"FAISS Vector Store created and saved at ./smiu-db")
        # faiss_vector = FAISS.load_local(
        #     "./smiu-db", 
        #     embeddings, 
        #     allow_dangerous_deserialization=True
        # )
        
        # query = "When was SMIU founded?"
        # search_results = faiss_vector.similarity_search(query, k=2)
        
        # print(f"\n🔍 Top 2 most relevant chunks for the query: '{query}'\n")
        # for i, result in enumerate(search_results, 1):
        #     print(f"Result {i}:")
        #     print(f"Content: {result.page_content}\n")
