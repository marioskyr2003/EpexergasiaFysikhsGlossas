
import os
import numpy as np
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import nltk

# Download NLTK resources (run once)
nltk.download('punkt')
nltk.download('stopwords')

# Load models
nlp = spacy.load("en_core_web_md")
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased")

gpt_pipeline = pipeline("text-generation", model="gpt2", framework="pt")

# File paths
base_path = "C:/Users/Mario/Desktop/EPEXERGASIAFYSIKHSGLWSSAS/"
paths = {
    "text1": os.path.join(base_path, "textone.txt"),
    "text2": os.path.join(base_path, "texttwo.txt"),
    "text1A": os.path.join(base_path, "out_textoneΑ.txt"),
    "text2A": os.path.join(base_path, "out_texttwoΑ.txt"),
    "text1B": os.path.join(base_path, "out_textoneB.txt"),
    "text2B": os.path.join(base_path, "out_texttwoB.txt"),
}

# Enhanced text preprocessing
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())
    
    # Remove punctuation and stopwords
    stop_words = set(stopwords.words('english') + list(string.punctuation))
    tokens = [token for token in tokens if token not in stop_words and token.isalpha()]
    
    return " ".join(tokens)

# Load plain text with preprocessing
def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    return preprocess_text(text)

# Extract B methods from the out_textB.txt
def extract_b_versions(text):
    sections = text.split("====")
    versions = {}
    for i in range(1, len(sections), 2):
        header = sections[i].strip().split()[0]
        body = "\n".join(sections[i + 1].strip().splitlines()[1:])
        versions[header] = preprocess_text(body)
    return versions

# Process text into vectors with enhanced options
def process_text(text, model="spacy"):
    if model == "spacy":
        doc = nlp(text)
        vectors = [token.vector for token in doc if token.has_vector]
        tokens = [token.text for token in doc if token.has_vector]
        return vectors, tokens
    elif model == "bert":
        inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        
        # Get all token embeddings (excluding [CLS] and [SEP])
        token_embeddings = outputs.last_hidden_state[0][1:-1]  # Remove first and last tokens
        
        # Convert to numpy array
        embeddings = token_embeddings.detach().numpy()
        tokens = bert_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][1:-1])
        
        # Filter out special tokens and get corresponding embeddings
        valid_tokens = []
        valid_embeddings = []
        for token, embedding in zip(tokens, embeddings):
            if token not in ['[CLS]', '[SEP]', '[PAD]'] and not token.startswith('##'):
                valid_tokens.append(token)
                valid_embeddings.append(embedding)
        
        return valid_embeddings, valid_tokens
    else:
        raise ValueError("Model must be either 'spacy' or 'bert'.")

# Mean vector calculation with proper reshaping for BERT
def get_avg_vector(vectors):
    if len(vectors) == 0:
        return np.zeros(300 if isinstance(vectors, list) else 768)  # spaCy vs BERT
    
    if isinstance(vectors, list):
        avg = np.mean(np.array(vectors), axis=0)
    else:
        avg = np.mean(vectors, axis=0)
    
    # Ensure 2D array for cosine similarity
    return avg.reshape(1, -1) if len(avg.shape) == 1 else avg

# GPT generation with enhanced parameters
def gpt_reconstruct(text):
    output = gpt_pipeline(
        text,
        max_new_tokens=100,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )[0]["generated_text"]
    return preprocess_text(output)

# Enhanced similarity calculations
def calculate_similarities(original_vectors, original_tokens, reconstructed_vectors, reconstructed_tokens):
    if len(original_vectors) == 0 or len(reconstructed_vectors) == 0:
        return 0.0, {}
    
    # Convert to numpy arrays if they aren't already
    orig_vecs = np.array(original_vectors)
    recon_vecs = np.array(reconstructed_vectors)
    
    # Word-level similarity
    word_similarities = {}
    similarity_matrix = cosine_similarity(orig_vecs, recon_vecs)
    
    # Find most similar words
    for i, orig_token in enumerate(original_tokens):
        if i >= len(similarity_matrix):
            continue
        most_sim_idx = np.argmax(similarity_matrix[i])
        word_similarities[orig_token] = {
            'most_similar': reconstructed_tokens[most_sim_idx],
            'score': float(similarity_matrix[i][most_sim_idx])  # Convert to Python float
        }
    
    avg_similarity = float(np.mean(np.max(similarity_matrix, axis=1)))  # Convert to Python float
    return avg_similarity, word_similarities

# Visualization function
def visualize_embeddings(original_vectors, original_tokens, reconstructed_vectors, reconstructed_tokens, method_name, model_name):
    # Combine vectors and labels
    all_vectors = np.vstack([np.array(original_vectors), np.array(reconstructed_vectors)])
    labels = original_tokens + reconstructed_tokens
    types = ['original'] * len(original_tokens) + ['reconstructed'] * len(reconstructed_tokens)
    
    # Apply dimensionality reduction
    pca = PCA(n_components=2)
    tsne = TSNE(n_components=2, perplexity=min(30, len(all_vectors)-1))
    
    reduced_pca = pca.fit_transform(all_vectors)
    reduced_tsne = tsne.fit_transform(all_vectors)
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # PCA plot
    for label_type in ['original', 'reconstructed']:
        indices = [i for i, t in enumerate(types) if t == label_type]
        ax1.scatter(reduced_pca[indices, 0], reduced_pca[indices, 1], label=label_type, alpha=0.6)
    ax1.set_title(f'PCA - {method_name} ({model_name})')
    ax1.legend()
    
    # t-SNE plot
    for label_type in ['original', 'reconstructed']:
        indices = [i for i, t in enumerate(types) if t == label_type]
        ax2.scatter(reduced_tsne[indices, 0], reduced_tsne[indices, 1], label=label_type, alpha=0.6)
    ax2.set_title(f't-SNE - {method_name} ({model_name})')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'embeddings_{method_name}_{model_name}.png')
    plt.close()

# Enhanced analysis core
def analyze_texts(original, reconstructed_versions, model="spacy"):
    original_vectors, original_tokens = process_text(original, model)
    original_avg = get_avg_vector(original_vectors)
    
    results = {}
    word_comparisons = {}
    
    for method, text in reconstructed_versions.items():
        recon_vectors, recon_tokens = process_text(text, model)
        recon_avg = get_avg_vector(recon_vectors)
        
        # Ensure both averages are 2D arrays
        orig_avg_2d = original_avg.reshape(1, -1) if len(original_avg.shape) == 1 else original_avg
        recon_avg_2d = recon_avg.reshape(1, -1) if len(recon_avg.shape) == 1 else recon_avg
        
        doc_similarity = float(cosine_similarity(orig_avg_2d, recon_avg_2d)[0][0])  # Convert to Python float
        word_similarity, word_sims = calculate_similarities(original_vectors, original_tokens, recon_vectors, recon_tokens)
        
        results[method] = {
            'document_similarity': doc_similarity,
            'word_similarity': word_similarity,
            'vector_difference': float(np.linalg.norm(original_avg - recon_avg)),  # Convert to Python float
            'word_comparisons': word_sims
        }
        
        # Visualize embeddings for the first few methods to avoid too many plots
        if method in list(reconstructed_versions.keys())[:3]:
            visualize_embeddings(
                original_vectors[:100], original_tokens[:100],  # Limit to 100 tokens for visualization
                recon_vectors[:100], recon_tokens[:100],
                method, model
            )
    
    return results, word_comparisons

# Enhanced main function with more detailed output
def main():
    analysis_results = {}
    
    for model in ["spacy", "bert"]:
        print(f"\n========= MODEL: {model.upper()} =========")
        model_results = {}
        
        for key in ["text1", "text2"]:
            original = load_text(paths[key])
            A_version = load_text(paths[key + "A"])
            B_versions = extract_b_versions(load_text(paths[key + "B"]))

            all_versions = {"A": A_version}
            all_versions.update(B_versions)
            all_versions["GPT"] = gpt_reconstruct(original)

            print(f"\n--- ANALYSIS FOR {key.upper()} ---")
            results, word_comparisons = analyze_texts(original, all_versions, model=model)
            model_results[key] = results
            
            # Print summary table
            print(f"\n{'Method':<15} {'Doc Similarity':<15} {'Word Similarity':<15} {'Vector Diff':<15}")
            for method, metrics in results.items():
                print(f"{method:<15} {metrics['document_similarity']:.4f}        {metrics['word_similarity']:.4f}        {metrics['vector_difference']:.4f}")
            
            # Print some example word comparisons
            print("\nExample word comparisons:")
            for method, metrics in results.items():
                if metrics['word_comparisons']:
                    example_words = list(metrics['word_comparisons'].items())[:3]  # First 3 examples
                    print(f"\nMethod {method}:")
                    for word, comparison in example_words:
                        print(f"  '{word}' → '{comparison['most_similar']}' (score: {comparison['score']:.2f})")
        
        analysis_results[model] = model_results
    
    return analysis_results

if __name__ == "__main__":
    full_results = main()
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_types(x) for x in obj]
        return obj
    
    # Save the full results to a file for further analysis
    import json
    with open('analysis_results.json', 'w') as f:
        json.dump(convert_types(full_results), f, indent=2)