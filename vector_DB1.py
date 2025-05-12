from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Create embeddings
embedding_model = OpenAIEmbeddings()

# Example documents
texts = ["apple is a fruit", "carrot is a vegetable", "python is a programming language"]

# Store vectors
vector_db = Chroma.from_texts(texts, embedding=embedding_model)

# Search
query = "what is a fruit?"
results = vector_db.similarity_search(query)

print(results[0].page_content)
