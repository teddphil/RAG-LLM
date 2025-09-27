import os
from typing import List, Any, Optional
from langchain_core.documents import Document
from langchain_core.language_models import BaseLLM
from langchain_core.embeddings import Embeddings
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.prompts import PromptTemplate
from langchain_core.outputs import LLMResult, Generation # <-- NEW IMPORT
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# --- 1. Mock Components for Local Runnability ---

class MockLLM(BaseLLM):
    """
    A mock LLM that simulates generating a response by confirming it received
    the prompt and the context retrieved from the RAG system.
    This avoids requiring a real LLM setup for the structural RAG demo.
    """
    @property
    def _llm_type(self) -> str:
        return "mock-llm"

    # Changed from _call to _generate to support newer LangChain versions
    def _generate(
        self,
        prompts: List[str], # LangChain's _generate takes a list of prompts
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> LLMResult:
        """The required abstract method for BaseLLM."""
        generations = []
        for prompt in prompts:
            # The prompt contains both the question and the retrieved context.
            if "Context:" in prompt:
                # Simple check to see if the RAG system successfully augmented the prompt
                response_text = f"Mock Response: I analyzed the retrieved context and the question. The RAG pipeline worked! (Full Prompt Received: {prompt[:100]}...)"
            else:
                response_text = f"Mock Response: I only received a question without context. (Prompt: {prompt[:100]}...)"
            
            # Wrap the text in the required Generation structure
            generations.append([Generation(text=response_text)])
            
        return LLMResult(generations=generations) # Return the required LLMResult object

class FakeEmbeddings(Embeddings):
    """
    A mock Embeddings class that simulates converting text to vectors.
    It returns a dummy vector of 1536 dimensions for every input.
    """
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Simulates a fixed-size vector output for the RAG index
        return [[1.0] * 1536 for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        # Simulates a fixed-size vector output for the query
        return [0.5] * 1536

# --- 2. Data and Initialization ---

def get_knowledge_base_data() -> List[Document]:
    """Provides the internal documents for the RAG knowledge base."""
    print("--- 1. Loading Internal Documents ---")
    
    # These are the proprietary documents/data your LLM should reference.
    data = [
        Document(
            page_content="The Q2 2024 revenue for the 'Nebula' product line reached $1.5 million, exceeding projections by 15%. This growth is attributed to the success of the new marketing campaign in the European sector.",
            metadata={"source": "Q2_Report_2024.txt", "category": "Finance"}
        ),
        Document(
            page_content="All employees are entitled to 20 days of paid vacation leave per calendar year. Sick leave is separate and provides up to 10 additional days. Vacation requests must be submitted through the internal HR portal 14 days in advance.",
            metadata={"source": "HR_Policy.pdf", "category": "HR"}
        ),
        Document(
            page_content="Our primary codebase is written in Python (80%) and TypeScript (20%). The core RAG component relies heavily on the LangChain framework for orchestration and FAISS for vector indexing.",
            metadata={"source": "Dev_Specs.md", "category": "Tech"}
        ),
        Document(
            page_content="The company vision is to become the leading provider of personalized, on-device AI solutions by 2027.",
            metadata={"source": "Company_Vision.txt", "category": "General"}
        ),
    ]
    print(f"Loaded {len(data)} documents for the knowledge base.")
    return data

def setup_rag_pipeline(data: List[Document]) -> RetrievalQA:
    """Sets up the RAG pipeline using LangChain components."""
    
    # 2. Split Documents
    print("--- 2. Splitting Documents ---")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    docs = text_splitter.split_documents(data)
    print(f"Documents split into {len(docs)} chunks.")

    # 3. Create Embeddings and Vector Store
    print("--- 3. Creating Vector Store (FAISS) ---")
    embeddings = FakeEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    print("Vector Store created and indexed (in-memory).")

    # 4. Initialize LLM and RAG Chain
    print("--- 4. Initializing RAG Chain ---")
    llm = MockLLM()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) # Retrieve top 2 chunks

    # Define the custom prompt template
    RAG_PROMPT_TEMPLATE = """
    You are a helpful data assistant. Use the following pieces of retrieved context 
    to answer the user's question. If you don't know the answer, just state that 
    you could not find enough relevant information.

    Question: {question} 

    Context:
    {context}

    Answer:
    """
    
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # Stuff all retrieved docs into the prompt
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)}
    )

    print("RAG pipeline setup complete.")
    return rag_chain

# --- 3. Main Execution ---

def main():
    """Main function to run the RAG Data Assistant demo."""
    
    # Initialize the RAG pipeline
    knowledge_base = get_knowledge_base_data()
    rag_chain = setup_rag_pipeline(knowledge_base)

    print("\n=======================================================")
    print(" RAG DATA ASSISTANT DEMO (Mock LLM) ")
    print("=======================================================")

    queries = [
        "What was the revenue for the Nebula product line in Q2 2024?",
        "How much vacation time are employees allowed?",
        "What is the company's primary vision for 2027?",
        "Tell me about our software development process."
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n--- QUERY {i}: {query} ---")
        
        # Execute the RAG chain
        result = rag_chain.invoke({"query": query})
        
        # Extract results
        answer = result["result"]
        source_docs = result["source_documents"]

        print(f"\n[ASSISTANT RESPONSE]: {answer}")
        
        print("\n[RETRIEVED SOURCES USED]:")
        for doc in source_docs:
            # We truncate the content for cleaner output in the console
            print(f"- Source: {doc.metadata['source']} | Content Snippet: {doc.page_content[:50]}...")
            
    print("\n=======================================================")
    print("DEMO COMPLETE. Modify the data and prompt for custom use.")
    print("=======================================================")


if __name__ == "__main__":
    main()
