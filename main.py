import os
import warnings
import sys

# === НАСТРОЙКА ПУТЕЙ ===
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_CACHE_PATH = os.path.join(PROJECT_ROOT, "models_cache")
DB_FOLDER = os.path.join(PROJECT_ROOT, "db_storage")
DOCS_FOLDER = os.path.join(PROJECT_ROOT, "docs")

os.makedirs(MODEL_CACHE_PATH, exist_ok=True)
os.makedirs(DB_FOLDER, exist_ok=True)

# === ОТКЛЮЧЕНИЕ ВСЕХ ПРЕДУПРЕЖДЕНИЙ ===
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

# === ПРАВИЛЬНЫЙ ИМПОРТ (без DeprecationWarning!) ===
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_documents(folder_path: str):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Папка {folder_path} не найдена.")

    loader = DirectoryLoader(
        folder_path,
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader,
        loader_kwargs={"extract_images": False}
    )
    documents = loader.load()
    print(f"✅ Загружено документов: {len(documents)}")
    return documents


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=300,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Создано чанков: {len(chunks)}")
    return chunks


def create_vector_store(chunks):
    print(f"⏳ Инициализация эмбеддингов... (кэш: {MODEL_CACHE_PATH})")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        cache_folder=MODEL_CACHE_PATH,
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_FOLDER
    )
    print(f"✅ База создана: {DB_FOLDER}")
    return vectorstore


def load_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        cache_folder=MODEL_CACHE_PATH,
        encode_kwargs={"normalize_embeddings": True}
    )
    return Chroma(
        persist_directory=DB_FOLDER,
        embedding_function=embeddings
    )


def initialize_db():
    if os.path.exists(DB_FOLDER) and os.listdir(DB_FOLDER):
        print("⏳ Загрузка существующей БД...")
        return load_vector_store()
    else:
        print("⏳ Создание новой БД...")
        docs = load_documents(DOCS_FOLDER)
        chunks = split_documents(docs)
        return create_vector_store(chunks)


def search_documents(query: str, vectorstore, k: int = 3):
    results = vectorstore.similarity_search(query, k=k)
    return results


if __name__ == "__main__":
    try:
        vectorstore = initialize_db()
        print("\n🎉 ГОТОВО! Введите запрос для поиска (или 'exit'):")

        while True:
            query = input("\n> ")
            if query.lower() == 'exit':
                break
            if not query.strip():
                continue

            docs = search_documents(query, vectorstore, k=3)
            if not docs:
                print("❌ Ничего не найдено.")
            else:
                for i, doc in enumerate(docs, 1):
                    src = os.path.basename(doc.metadata.get('source', '?'))
                    page = doc.metadata.get('page', '?')
                    content = doc.page_content[:250].replace('\n', ' ')
                    print(f"{i}. [{src}:стр.{page}] {content}...")

    except KeyboardInterrupt:
        print("\n👋 Прервано.")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}", file=sys.stderr)