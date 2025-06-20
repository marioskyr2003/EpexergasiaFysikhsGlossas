
#  Επεξεργασία Φυσικής Γλώσσας – Απαλλακτική Εργασία
**Μάριος Κυρόγλου – Π21080**

## Περιγραφή
Απαλλακτική εργασία για το μάθημα Επεξεργασία Φυσικής Γλώσσας (Παν. Πειραιώς). Περιλαμβάνει παραδοτέα με στόχο την ανάλυση, παραφραστική αναδιατύπωση και αξιολόγηση κειμένων με σύγχρονες τεχνικές NLP.

## Δομή Φακέλων

```
EFG-teliko/
├── THEMA 1/              # Παραδοτέο 1: Βασική παραφραστική μέθοδος
├── THEMA 2/              # Παραδοτέο 2: GPT-based παραφράσεις, BERT/SpaCy embeddings
```

##  Τρόπος Εκτέλεσης

1. **Δημιουργία virtual environment  μέσω Anaconda**
2. **Εγκατάσταση  requirements**:
   ```bash
   pip install -r requirements.txt
   ```

3. **μοντέλα NLP**:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('averaged_perceptron_tagger')

   # Για textblob
   python -m textblob.download_corpora

   # Για spaCy
   python -m spacy download en_core_web_md
   python -m spacy download en_core_web_sm
   ```



##  Περιεχόμενο Παραδοτέων

- **THEMA 1**: Αντικατάσταση συνωνύμων με NLTK & TextBlob + documentation.
- **THEMA 2**: Παραφράσεις με GPT-2, υπολογισμός ομοιότητας με SpaCy/BERT embeddings + documentation.


##  Απαιτήσεις

- Python 3.8+
- `anaconda enviroment`,`spaCy`, `transformers`, `nltk`, `textblob`, `sentence-transformers`, `torch`, `matplotlib`, `scikit-learn`

##  Τεκμηρίωση

THEMA 1/ Documentation - Παραδοτέο 1.pdf
THEMA 1/ Documentation - Παραδοτέο 2.pdf
