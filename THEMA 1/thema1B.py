# Import libraries
import numpy as np
import random
import os
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import wordnet
from gensim.models import Word2Vec
from itertools import chain
import spacy
nlp = spacy.load("en_core_web_sm")
from textblob import TextBlob

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Set seed for reproducibility
def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)

# NLTK paraphrasing
def nltk_paraphrase(tokens, replace_prob=0.3):
    new_tokens = []
    for token in tokens:
        if random.random() < replace_prob:
            synsets = wordnet.synsets(token)
            synonyms = set(chain.from_iterable([syn.lemma_names() for syn in synsets]))
            synonyms.discard(token)
            new_token = random.choice(list(synonyms)) if synonyms else token
        else:
            new_token = token
        new_tokens.append(new_token)
    return new_tokens

# spaCy paraphrasing
def spacy_paraphrase(text, replace_prob=0.3):
    doc = nlp(text)
    new_tokens = []
    for token in doc:
        if random.random() < replace_prob and token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']:
            synonyms = []
            for syn in wordnet.synsets(token.text):
                for lemma in syn.lemmas():
                    if lemma.name() != token.text.lower():
                        synonyms.append(lemma.name().replace('_', ' '))
            new_token = random.choice(synonyms) if synonyms else token.text
        else:
            new_token = token.text
        new_tokens.append(new_token)
    return ' '.join(new_tokens)

# TextBlob paraphrasing
def textblob_paraphrase(text):
    blob = TextBlob(text)
    paraphrased = []
    for word, tag in blob.tags:
        pos = None
        if tag.startswith('NN'): pos = 'n'
        elif tag.startswith('VB'): pos = 'v'
        elif tag.startswith('JJ'): pos = 'a'
        elif tag.startswith('RB'): pos = 'r'

        if pos:
            synsets = wordnet.synsets(word, pos=pos)
            if synsets:
                lemmas = set()
                for syn in synsets:
                    for lemma in syn.lemmas():
                        if lemma.name() != word.lower():
                            lemmas.add(lemma.name())
                if lemmas:
                    paraphrased.append(random.choice(list(lemmas)))
                    continue
        paraphrased.append(word)
    return ' '.join(paraphrased)

# Sentence embedding generation
def get_embeddings(text, model=None, method='nltk'):
    if method == 'nltk':
        tokens = word_tokenize(text.lower())
        vectors = [model.wv[token] for token in tokens if token in model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(100)
    elif method == 'spacy':
        return nlp(text).vector
    else:  # textblob
        return np.mean([nlp(str(word)).vector for word in TextBlob(text).words], axis=0)

# Analyze entire text file
def analyze_full_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    sentences = sent_tokenize(text)
    words_per_sentence = [word_tokenize(sent.lower()) for sent in sentences]
    w2v_model = Word2Vec(sentences=words_per_sentence, vector_size=100, window=5, min_count=1, workers=4, seed=42)

    full_text_original = ' '.join(sentences)
    nltk_reconstructed = ' '.join([' '.join(nltk_paraphrase(word_tokenize(sent))) for sent in sentences])
    spacy_reconstructed = ' '.join([spacy_paraphrase(sent) for sent in sentences])
    textblob_reconstructed = ' '.join([textblob_paraphrase(sent) for sent in sentences])

    # Compute similarities
    similarities = {
        'NLTK': cosine_similarity(
            [get_embeddings(full_text_original, w2v_model, 'nltk')],
            [get_embeddings(nltk_reconstructed, w2v_model, 'nltk')]
        )[0][0],
        'spaCy': cosine_similarity(
            [get_embeddings(full_text_original, method='spacy')],
            [get_embeddings(spacy_reconstructed, method='spacy')]
        )[0][0],
        'TextBlob': cosine_similarity(
            [get_embeddings(full_text_original, method='textblob')],
            [get_embeddings(textblob_reconstructed, method='textblob')]
        )[0][0],
    }

    paraphrased_texts = {
        'NLTK': nltk_reconstructed,
        'spaCy': spacy_reconstructed,
        'TextBlob': textblob_reconstructed
    }

    return {
        'original_text': full_text_original,
        'paraphrased_texts': paraphrased_texts,
        'similarities': similarities
    }

# Save paraphrased output to file
def save_outputs(result, original_path):
    base_name = os.path.basename(original_path)
    if base_name == "textone.txt":
        out_name = "out_textoneB.txt"
    elif base_name == "texttwo.txt":
        out_name = "out_texttwoB.txt"
    else:
        out_name = "out_unknownB.txt"

    folder = os.path.dirname(original_path)
    out_path = os.path.join(folder, out_name)

    with open(out_path, "w", encoding="utf-8") as f:
        for method in ['NLTK', 'spaCy', 'TextBlob']:
            f.write(f"==== {method} Paraphrased Version ====\n")
            f.write(result['paraphrased_texts'][method] + "\n\n")
            f.write(f"{method} Cosine Similarity: {result['similarities'][method]:.4f}\n\n")
    print(f">>> Paraphrased outputs saved to: {out_path}")

# MAIN execution
if __name__ == "__main__":
    set_seed(42)

    file_paths = [
        "C:/Users/Mario/Desktop/EPEXERGASIAFYSIKHSGLWSSAS/textone.txt",
        "C:/Users/Mario/Desktop/EPEXERGASIAFYSIKHSGLWSSAS/texttwo.txt"
    ]

    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        print(f"\n{'='*100}")
        print(f"FULL TEXT ANALYSIS FOR: {file_name}")
        print(f"{'='*100}")

        result = analyze_full_text(file_path)

        print("\nOriginal Text:\n", result['original_text'][:500], "...\n")

        for method in ['NLTK', 'spaCy', 'TextBlob']:
            print(f"\n{method} RECONSTRUCTED TEXT:\n{result['paraphrased_texts'][method][:500]} ...")
            print(f"{method} Cosine Similarity: {result['similarities'][method]:.4f}")

        # Save outputs to files
        save_outputs(result, file_path)
