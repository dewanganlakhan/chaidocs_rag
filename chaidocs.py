# pip install qdrant-client langchain openai tiktoken beautifulsoup4 requests
import os
from typing import List, Dict, Optional, Tuple
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models
from bs4 import BeautifulSoup
import requests
import re
from dotenv import load_dotenv
import os
    
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = "http://localhost:6333"  # or your Qdrant cloud URL
QDRANT_API_KEY = None  # or your Qdrant API key
COLLECTION_PREFIX = "chaidocs_"
DOCS_BASE_URL = "https://chaidocs.vercel.app"

# Initialize clients
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model="gpt-4", openai_api_key=OPENAI_API_KEY)
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

class ChaidocsRAG:
    def __init__(self):
        self.vector_stores = {}
        self.course_structure = self.parse_sidebar_structure()
        self.initialize_vector_stores()
    
    def parse_sidebar_structure(self) -> Dict[str, Dict[str, str]]:
        """Parse the sidebar HTML to extract course structure with all links"""
        try:
            response = requests.get(f"{DOCS_BASE_URL}/youtube/getting-started")
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find the sidebar navigation
            sidebar = soup.find('nav', {'aria-label': 'Main'}) or soup.find('nav')
            if not sidebar:
                raise ValueError("Could not find sidebar navigation")
            
            # Extract course structure
            course_structure = {}
            
            # Find all top-level list items containing course details
            top_level_items = sidebar.select('.top-level > li')
            
            for item in top_level_items:
                # Check if it's a course (has details element)
                details = item.find('details')
                if details:
                    # Extract course name
                    course_name = details.find('span', class_='large').get_text().strip()
                    
                    # Extract all links under this course
                    course_links = {}
                    for link in details.select('ul a[href]'):
                        page_name = link.get_text().strip()
                        page_url = f"{DOCS_BASE_URL}{link['href']}"
                        course_links[page_name] = page_url
                    
                    course_structure[course_name] = course_links
            
            return course_structure
        
        except Exception as e:
            print(f"Error parsing sidebar structure: {e}")
            # Fallback to default structure if parsing fails
            return {
                "Chai aur HTML": {
                    "Welcome": f"{DOCS_BASE_URL}/youtube/chai-aur-html/welcome/",
                    "HTML Intro": f"{DOCS_BASE_URL}/youtube/chai-aur-html/introduction/",
                    "Emmet Crash Course": f"{DOCS_BASE_URL}/youtube/chai-aur-html/emmit-crash-course/",
                    "Common HTML Tags": f"{DOCS_BASE_URL}/youtube/chai-aur-html/html-tags/"
                },
                "Chai aur Git": {
                    "Welcome": f"{DOCS_BASE_URL}/youtube/chai-aur-git/welcome/",
                    "Git and GitHub": f"{DOCS_BASE_URL}/youtube/chai-aur-git/introduction/",
                    "Terminology": f"{DOCS_BASE_URL}/youtube/chai-aur-git/terminology/",
                    "Behind the scenes": f"{DOCS_BASE_URL}/youtube/chai-aur-git/behind-the-scenes/"
                }
            }
    
    def initialize_vector_stores(self):
        """Initialize Qdrant collections for each detected course"""
        for course_name in self.course_structure.keys():
            # Create collection name by removing spaces and special chars
            collection_name = f"{COLLECTION_PREFIX}{re.sub(r'[^a-zA-Z0-9]', '_', course_name).lower()}"
            self.vector_stores[course_name] = Qdrant(
                client=qdrant_client,
                collection_name=collection_name,
                embeddings=embeddings
            )
    
    def load_and_chunk_course_docs(self) -> Dict[str, List[Document]]:
        """Load all course documents and split into chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        course_docs = {}
        
        for course_name, pages in self.course_structure.items():
            try:
                print(f"\nProcessing course: {course_name}")
                course_chunks = []
                
                for page_name, page_url in pages.items():
                    try:
                        print(f"  Loading page: {page_name} from {page_url}")
                        loader = WebBaseLoader(page_url)
                        docs = loader.load()
                        
                        # Add metadata to each document
                        for doc in docs:
                            doc.metadata.update({
                                "source": page_url,
                                "course": course_name,
                                "page": page_name
                            })
                        
                        # Split into chunks
                        chunks = text_splitter.split_documents(docs)
                        course_chunks.extend(chunks)
                        
                    except Exception as e:
                        print(f"  Error loading page {page_name}: {e}")
                
                course_docs[course_name] = course_chunks
            
            except Exception as e:
                print(f"Error processing course {course_name}: {e}")
        
        return course_docs
    
    def index_documents(self):
        """Index documents into their respective Qdrant collections"""
        course_docs = self.load_and_chunk_course_docs()
        
        for course_name, docs in course_docs.items():
            if docs:
                # Create collection name by removing spaces and special chars
                collection_name = f"{COLLECTION_PREFIX}{re.sub(r'[^a-zA-Z0-9]', '_', course_name).lower()}"
                
                # Create collection if it doesn't exist
                try:
                    qdrant_client.get_collection(collection_name)
                except:
                    qdrant_client.create_collection(
                        collection_name=collection_name,
                        vectors_config=models.VectorParams(
                            size=1536,  # OpenAI embedding size
                            distance=models.Distance.COSINE
                        )
                    )
                
                # Add documents to collection
                self.vector_stores[course_name].add_documents(docs)
                print(f"Indexed {len(docs)} chunks for course: {course_name}")
    
    def determine_relevant_course(self, query: str) -> Tuple[str, float]:
        """Use LLM to determine which course is most relevant to the query"""
        prompt = ChatPromptTemplate.from_template(
            """Based on the user's question, determine which Chaidocs course is most relevant. 
            The available courses are:
            {courses}
            
            Question: {question}
            
            Respond with ONLY the course name exactly as shown, nothing else."""
        )
        
        chain = (
            {
                "courses": lambda x: "\n- ".join(self.course_structure.keys()),
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        
        course = chain.invoke(query).strip()
        return course if course in self.course_structure else None
    
    def query_docs(self, question: str) -> dict:
        """Query the documentation with routing to the appropriate course"""
        # First determine which course to query
        course = self.determine_relevant_course(question)
        
        if not course:
            return {
                "answer": "I couldn't determine which course documentation to search.",
                "sources": []
            }
        
        # Search in the appropriate collection
        try:
            retriever = self.vector_stores[course].as_retriever(search_kwargs={"k": 5})
            docs = retriever.get_relevant_documents(question)
        except KeyError:
            return {
                "answer": f"The course '{course}' was identified but not found in the database.",
                "sources": []
            }
        
        # Format the context for the LLM
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Get unique sources with their page names
        source_info = {}
        for doc in docs:
            source_url = doc.metadata["source"]
            page_name = doc.metadata["page"]
            if source_url not in source_info:
                source_info[source_url] = page_name
        
        # Generate answer with sources
        prompt = ChatPromptTemplate.from_template(
            """Answer the question based only on the following context from the {course} course:
            {context}
            
            Question: {question}
            
            Include the relevant documentation link(s) in your answer.
            If the answer isn't in the context, say you don't know."""
        )
        
        chain = (
            {
                "context": lambda x: context, 
                "question": RunnablePassthrough(),
                "course": lambda x: course
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        
        answer = chain.invoke(question)
        
        # Add sources at the end if they're not already mentioned
        if not any(source in answer for source in source_info):
            sources_text = "\n\nSources:\n" + "\n".join(
                [f"- {page}: {url}" for url, page in source_info.items()]
            )
            answer += sources_text
        
        return {
            "answer": answer,
            "sources": source_info,
            "course": course
        }

# Usage example
if __name__ == "__main__":
    print("Initializing Chaidocs RAG system...")
    rag = ChaidocsRAG()
    
    # Index documents (do this once)
    print("\nIndexing documents...")
    rag.index_documents()
    
    # Example questions
    example_questions = [
        "How do I use Emmet in HTML?",
        "What are Git branches and how do they work?",
        "Explain PostgreSQL database design best practices",
        "How to set up Nginx rate limiting?"
    ]
    
    print("\nTry these example questions:")
    for i, question in enumerate(example_questions, 1):
        print(f"{i}. {question}")
    
    # Interactive query loop
    while True:
        question = input("\nAsk a question about Chaidocs (or 'quit' to exit): ")
        if question.lower() == 'quit' or question.lower() == 'exit':
            break
            
        result = rag.query_docs(question)
        print("\nAnswer:")
        print(result["answer"])
        print(f"\nCourse searched: {result['course']}")
