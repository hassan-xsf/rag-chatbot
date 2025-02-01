from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from concurrent.futures import ThreadPoolExecutor

# Initialize variables
collection_name = "smiu-data"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def load_data():
    try:
        current_dir = Path.cwd()
        data_path = current_dir / "data" / "smiu-data.md"
        with open(data_path, "r", encoding="utf-8") as file:
            text_content = file.read()
        return text_content
    except FileNotFoundError:
        print("Error: Data file not found")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def embed_in_parallel(splits, batch_size=50):
    """ Simple batch embedding using parallelism """
    batches = [splits[i:i + batch_size] for i in range(0, len(splits), batch_size)]
    document_embeddings = []
    
    with ThreadPoolExecutor() as executor:
        # Embedding each batch in parallel
        embeddings_chunks = list(executor.map(embeddings.embed_documents, batches))
    
    # Flatten the list of embeddings
    for embeddings_chunk in embeddings_chunks:
        document_embeddings.extend(embeddings_chunk)
    
    return document_embeddings

if __name__ == "__main__":
    text = load_data()
    if text:
        print("Data loaded successfully")

        # Split the text into smaller chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Adjust chunk size for better performance
            chunk_overlap=200,
            length_function=len
        )
        splits = splitter.split_text(text)
        print(f"Split the file into {len(splits)} chunks.")

        # Generate embeddings in parallel using batch processing
        document_embeddings = embed_in_parallel(splits)
        print(f"Created embeddings for {len(document_embeddings)} document chunks.")

        # Create the vector store
        chrome_vector = Chroma.from_documents(
            collection_name=collection_name,
            documents=document_embeddings,
            embeddings=embeddings,
            persist_directory="./smiu-db"
        )
        print(f"Vector Store for collection {collection_name} has been created at ./smiu-db")

        # Perform a similarity search
        query = "When was SMIU founded?"
        search_results = chrome_vector.similarity_search(query, k=2)
        
        # Display top results
        print(f"\nTop 2 most relevant chunks for the query: '{query}'\n")
        for i, result in enumerate(search_results, 1):
            print(f"Result {i}:")
            print(f"Source: {result.metadata.get('source', 'Unknown')}")
            print(f"Content: {result.page_content}")
