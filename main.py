import os
import shutil
import time
import logging
from typing import List
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RAGSystem:
    def __init__(self, persist_dir="./chroma_db", docs_dir="documents/"):
        load_dotenv()
        self.google_api_key = os.getenv("API_KEY")
        self.persist_dir = persist_dir
        self.docs_dir = docs_dir

        # 1. Initialize Embeddings & Model
        # Using a reliable local embedding model
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Note: Ensure your environment supports Gemini 2.5 or adjust to "gemini-1.5-flash"
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=self.google_api_key,
            temperature=0,
            max_retries=6
        )

        self.vectorstore = None
        self.retriever = None

        # 2. Automatically attempt to load existing DB on startup
        self._bootstrap_db()

    def _bootstrap_db(self):
        """Checks if the vector store already exists and loads it."""
        if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir):
            logger.info("Existing database found. Loading...")
            self.vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings
            )
            self._setup_retriever()
        else:
            logger.info("No existing database found. System is idle.")

    def _setup_retriever(self):
        """Helper to configure retriever settings."""
        if self.vectorstore:
            self.retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 3, "fetch_k": 10}
            )

    def _clear_db(self):
        """Safely clears existing vector database."""
        if os.path.exists(self.persist_dir):
            logger.info("Clearing existing database.")
            try:
                shutil.rmtree(self.persist_dir)
                # Small sleep to ensure OS releases file locks
                time.sleep(1)
            except Exception as e:
                logger.error(f"Failed to clear directory: {e}")

    def load_and_index(self, batch_size=50):
        """Loads documents and indexes them with manual directory handling."""
        if not os.path.exists(self.docs_dir):
            logger.error(f"Directory '{self.docs_dir}' not found.")
            return

        # 1. Clear and Recreate Directory Manually
        if os.path.exists(self.persist_dir):
            try:
                shutil.rmtree(self.persist_dir)
                time.sleep(2)  # Give the OS time to release locks
            except Exception as e:
                logger.error(f"Error clearing DB: {e}")

        os.makedirs(self.persist_dir, exist_ok=True)

        # 2. Document Loading
        loaders = {
            ".pdf": (PyPDFLoader, {}),
            ".txt": (TextLoader, {"encoding": "utf-8"}),
        }

        docs = []
        for ext, (loader_cls, kwargs) in loaders.items():
            try:
                loader = DirectoryLoader(self.docs_dir, glob=f"*{ext}", loader_cls=loader_cls, loader_kwargs=kwargs)
                docs.extend(loader.load())
            except Exception as e:
                logger.warning(f"Error loading {ext} files: {e}")

        if not docs:
            logger.error("No documents found.")
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)

        # 3. Explicit Initialization
        # Instead of from_documents, we initialize the object first
        try:
            self.vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings
            )

            # Add in batches
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i: i + batch_size]
                self.vectorstore.add_documents(documents=batch)
                logger.info(f"Progress: {i + len(batch)}/{len(chunks)} chunks indexed.")

        except Exception as e:
            logger.error(f"Chroma initialization failed: {e}")
            return f"Error: {str(e)}"

        self._setup_retriever()
        logger.info("Indexing complete.")

    def format_docs_with_metadata(self, docs):
        """Formats docs to include source information."""
        formatted = []
        for doc in docs:
            source = os.path.basename(doc.metadata.get('source', 'Unknown'))
            formatted.append(f"[Source: {source}]\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted)

    def get_chain(self):
        """Builds the LCEL Chain."""
        template = """Answer the question based ONLY on the following context. 
        If the answer isn't in the context, say you don't know.

        Context:
        {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        return (
                {"context": self.retriever | self.format_docs_with_metadata, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
        )

    def ask(self, query: str):
        """Executes query with verification that the DB is loaded."""
        if not self.retriever:
            return "Knowledge base is empty. Please place PDFs in the 'documents' folder and click 'Refresh Knowledge Base'."

        try:
            chain = self.get_chain()
            return chain.invoke(query)
        except Exception as e:
            logger.error(f"RAG Chain Error: {e}")
            return "I'm sorry, I encountered an error while processing your request."