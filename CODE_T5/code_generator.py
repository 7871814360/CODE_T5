import streamlit as st
import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Check if GPU (CUDA) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Using device: {device}")

# Load the CSV file
data = pd.read_csv('python_code.csv')

# Load the question and solution columns
questions = data['question'].tolist()
solutions = data['solution'].tolist()

# Step 1: Encode the questions into embeddings using a sentence transformer
embedder_model = SentenceTransformer('all-mpnet-base-v2')
question_embeddings = embedder_model.encode(questions, convert_to_tensor=True)

# Step 2: Create FAISS index for retrieval
d = question_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(np.array(question_embeddings.cpu()))

SIMILARITY_THRESHOLD = 0.5

def retrieve_answer(input_question):
    input_embedding = embedder_model.encode([input_question], convert_to_tensor=True)
    input_embedding = np.array(input_embedding.cpu())
    D, I = index.search(input_embedding, k=1)
    distance = D[0][0]

    if distance < SIMILARITY_THRESHOLD:
        return solutions[I[0][0]], questions[I[0][0]] 
    else:
        return None, None 

# Load the quantized CodeT5 model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-large-ntp-py")
codet5_model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-large-ntp-py")

def generate_code(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generated_tokens = codet5_model.generate(**inputs, max_length=200)
    return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

def rag_generate(input_question):
    retrieved_solution, matched_question = retrieve_answer(input_question)
    if retrieved_solution is not None:
        prompt = f"# Input Question: {input_question}\n# Retrieved Solution: {retrieved_solution}\n"
        generated_code = generate_code(prompt)
        return {
            'input_question': input_question,
            'matched_question': matched_question,
            'retrieved_solution': retrieved_solution,
            'generated_code': generated_code
        }
    else:
        return {
            'input_question': input_question,
            'error': "No relevant questions found."
        }

# Streamlit UI
st.title("Code Generation Assistant")
st.write("Enter your programming question below:")

user_question = st.text_input("Programming Question")

if st.button("Generate Code"):
    if user_question:
        result = rag_generate(user_question)
        if 'error' not in result:
            st.write(f"**Input Question:** {result['input_question']}")
            st.write(f"**Matched Question:** {result['matched_question']}")
            st.write(f"**Retrieved Solution:** {result['retrieved_solution']}")
            st.write("**Generated Code:**")
            st.code(result['generated_code'])
        else:
            st.error(result['error'])
    else:
        st.warning("Please enter a question to generate code.")