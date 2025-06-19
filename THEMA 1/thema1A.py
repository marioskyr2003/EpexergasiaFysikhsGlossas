# Import necessary libraries
import nltk
import random
import numpy as np
import os

from itertools import chain
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import wordnet
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# Ensure necessary NLTK packages are downloaded
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Reproducibility settings
def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)

set_seed(42)

# Read input text files
file_paths = [
    "C:/Users/Desktop/EPEXERGASIAFYSIKHSGLWSSAS/textone.txt",
    "C:/UsersDesktop/EPEXERGASIAFYSIKHSGLWSSAS/texttwo.txt"
]

all_results = []

for file_path in file_paths:
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Tokenize into sentences using sent_tokenize()
    sentences = sent_tokenize(text)
    words_per_sentence = [word_tokenize(sent) for sent in sentences]

    # Train a local Word2Vec model on the sentences
    w2v_model = Word2Vec(
        sentences=words_per_sentence,
        vector_size=100,
        window=5,
        min_count=1,
        workers=4,
        seed=42
    )

    # Function to compute average embedding of a sentence
    def sentence_embedding(tokens, model):
        vectors = [model.wv[token] for token in tokens if token in model.wv]
        if not vectors:
            return np.zeros(model.vector_size)
        return np.mean(vectors, axis=0)

    # Function to paraphrase a sentence by replacing words with synonyms
    def paraphrase_sentence(tokens, replace_prob=0.3):
        new_tokens = []
        for token in tokens:
            if random.random() < replace_prob:
                synsets = wordnet.synsets(token)
                lemma_names = set(chain.from_iterable([syn.lemma_names() for syn in synsets]))
                lemma_names.discard(token)
                new_token = random.choice(list(lemma_names)) if lemma_names else token
            else:
                new_token = token
            new_tokens.append(new_token)
        return new_tokens

    # Select two random sentences for paraphrasing
    target_indices = random.sample(range(len(sentences)), 2)
    results = []

    # Process each selected sentence
    for idx in target_indices:
        original_sentence = sentences[idx]
        original_tokens = word_tokenize(original_sentence)
        # Generate paraphrased version
        paraphrased_tokens = paraphrase_sentence(original_tokens)
        paraphrased_sentence = ' '.join(paraphrased_tokens)
        # Compute embeddings
        emb_original = sentence_embedding(original_tokens, w2v_model)
        emb_paraphrased = sentence_embedding(paraphrased_tokens, w2v_model)
        # Calculate cosine similarity
        cos_sim = cosine_similarity([emb_original], [emb_paraphrased])[0][0]
        results.append({
            'original': original_sentence,
            'paraphrased': paraphrased_sentence,
            'cosine_similarity': cos_sim
        })

    # Save paraphrased sentences to corresponding output file
    filename = os.path.basename(file_path)
    if filename == 'textone.txt':
        out_name = 'out_textoneΑ.txt'
    elif filename == 'texttwo.txt':
        out_name = 'out_texttwoΑ.txt'
    else:
        out_name = 'out_paraphrased.txt'  # fallback

    paraphrased_sentences = [r['paraphrased'] for r in results]
    output_text = '\n'.join(paraphrased_sentences)
    output_path = os.path.join(os.path.dirname(file_path), out_name)
    with open(output_path, 'w', encoding='utf-8') as out_f:
        out_f.write(output_text)

    all_results.append({
        'file_name': filename,
        'results': results
    })

# Print the analysis results
print("Paraphrased Sentence Analysis:\n")
for file_data in all_results:
    print(f"\n=== Results for {file_data['file_name']} ===")
    for item in file_data['results']:
        print(f"\nOriginal: {item['original']}")
        print(f"Paraphrased: {item['paraphrased']}")
        print(f"Cosine Similarity: {item['cosine_similarity']:.4f}")
