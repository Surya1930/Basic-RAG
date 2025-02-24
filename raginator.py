from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import google.generativeai as genai
from google.api_core import client_options as client_options_lib

your_api_key = os.getenv("API_KEY") ## remove this line while placing your api key down there

os.environ["GOOGLE_API_KEY"] = your_api_key
genai.configure(
    api_key=os.getenv("GOOGLE_API_KEY"),
    transport="rest",
    client_options=client_options_lib.ClientOptions()
)
model_flash = genai.GenerativeModel('gemini-1.5-flash-001')

def flash(prompt, model=model_flash, temperature = 0.5):
    try: 
        return model.generate_content(prompt, generation_config={'temperature': temperature})
    except Exception as e:
        print("Error generating response:", e)
        return None

model = SentenceTransformer('all-MiniLM-L6-v2')

code_folder = "code_rants"
code_dict = {}
for root, _, files in os.walk(code_folder):
    for file in files:
        if file.endswith(('.py','.cpp', '.txt')):
            path = os.path.relpath(os.path.join(root, file), code_folder)
            with open(path, 'r') as f:
                code_dict[path] = f.read()

texts = list(code_dict.values())
file_names = list(code_dict.keys())

vectors = model.encode(texts)

dimension = vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(vectors)
print(f"Loaded {len(texts)} code files into RAGinator")

query = "What's the loop code here?"
query_vector = model.encode([query])
D, I = index.search(query_vector, k=2)
print("\nTop matches:")

context = ""

for idx in I[0]:
    snippet = f"- From {file_names[idx]}: \n{texts[idx]}\n"
    print(snippet)
    context += snippet

rant_prompt = f"Rant about this code in Hitchhiker's Guide to the Galaxy style:\n{context}"
response = flash(rant_prompt, temperature=1.0)
if response:
    print("\nRAGinator Cosmic Rant:")
    print(response.text)
else:
    print("Rant failed-blame the Vogons!")