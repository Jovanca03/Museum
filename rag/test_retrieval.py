import os
from pathlib import Path
from dotenv import load_dotenv

# Kita buang import langchain.chains yang bermasalah
from langchain_chroma import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

CHROMA_DB_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma_db"

def format_docs(docs):
    """Fungsi pembantu untuk merangkum teks dari database."""
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain():
    # 1. Buka Vector Store
    embedding_function = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    vector_store = Chroma(
        persist_directory=str(CHROMA_DB_DIR),
        embedding_function=embedding_function,
        collection_name="museum_cakraningrat"
    )
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    # 2. Siapkan LLM Gemini
    # Mengambil nama model dari .env, dengan 3.6-flash sebagai default
    nama_model = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.6-flash")
    
    llm = ChatGoogleGenerativeAI(
        model=nama_model, 
        temperature=0.3
    )
    
    # 3. Prompt Sistem Avatar
    system_prompt = (
        "Anda adalah asisten virtual (Avatar Guide) untuk Museum Cakraningrat. "
        "Gunakan potongan konteks berikut untuk menjawab pertanyaan pengunjung. "
        "Jika Anda tidak tahu jawabannya berdasarkan konteks, katakan saja bahwa "
        "informasi tersebut belum tersedia di database museum. Jangan mengarang jawaban.\n\n"
        "Konteks:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # 4. Chain LCEL Baru (Menggantikan create_retrieval_chain)
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def main():
    print("Menginisialisasi Avatar Guide Museum Cakraningrat...")
    
    try:
        rag_chain = get_rag_chain()
    except Exception as e:
        print(f"Gagal menginisialisasi RAG: {e}")
        return

    print("\n" + "="*50)
    print("Sistem Siap! Silakan tanya seputar koleksi museum.")
    print("Ketik 'keluar' untuk berhenti.")
    print("="*50 + "\n")
    
    while True:
        pertanyaan = input("Pengunjung: ")
        
        if pertanyaan.lower() in ['keluar', 'exit', 'quit']:
            break
            
        if not pertanyaan.strip():
            continue
            
        print("Memikirkan jawaban...")
        
        # Eksekusi sistem RAG
        response = rag_chain.invoke(pertanyaan)
        
        print(f"\nAvatar Guide: {response}\n")

if __name__ == "__main__":
    main()