from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
import google.generativeai as genai

# Configure Google API Key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)

# Load the PDF
# Get the directory where the current script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build the full path to the PDF
pdf_path = os.path.join(script_dir, "Rideshare_Payload_Users_Guide.pdf")
try:
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
except FileNotFoundError:
    raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

# Split the documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# Embed the documents using Google's text-embedding-004
try:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vectorstore = FAISS.from_documents(docs, embeddings)
except Exception as e:
    raise Exception(f"Error creating embeddings: {e}")

# Use Gemini 1.5 Flash as the LLM (this is the free tier)
try:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
except Exception as e:
    raise Exception(f"Error initializing gemini-1.5-flash: {e}")

# Combine into a RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# Prompt
query = "What are the customer deliverables and verification requirements for payload integration in the SpaceX Rideshare Payload User’s Guide?"
try:
    result = qa_chain.invoke({"query": query})
except Exception as e:
    raise Exception(f"Error invoking QA chain: {e}")

# Display the results
print("Answer:\n", result['result'])
print("\n--- Source Documents ---")
for doc in result['source_documents']:
    print(doc.metadata.get('source', 'Unknown source'))