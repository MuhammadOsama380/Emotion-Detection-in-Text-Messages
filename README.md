# Emotion Detection in Text Messages

**Author:** Muhammad Osama  
**Student ID:** 1288056  
**Course:** Natural Language Processing with Python  
**Institution:** Fanshawe College  
**LinkedIn:** [muhammad-osama-872328202](https://www.linkedin.com/in/muhammad-osama-872328202)  
**GitHub:** [MuhammadOsama380](https://github.com/MuhammadOsama380)

---

## Project Overview
This project focuses on detecting **emotions from text messages** using natural language processing and classical machine learning techniques.  
The dataset consists of Twitter messages categorized into six emotions — **sadness (0), joy (1), love (2), anger (3), fear (4), and surprise (5)**.

The goal is to compare different **text preprocessing and embedding techniques** — including **One-Hot Encoding**, **TF-IDF**, and **Word2Vec** — to determine which best captures emotional patterns in text.

---

## Dataset
- **Source:** [Kaggle - Emotion Dataset](https://www.kaggle.com/datasets/parulpandey/emotion-dataset)
- **Records:** 2,000  
- **Columns:**  
  - `text` → message text  
  - `label` → emotion category (0–5)
- **Emotions Covered:** sadness, joy, love, anger, fear, surprise  

Dataset was collected from English Twitter messages and labeled using emotion-specific hashtags.

---

## Exploratory Data Analysis
### Visualizations
- **Label Distribution:** Bar chart showing class imbalance — sadness and joy dominate the dataset.  
- **Message Length Distribution:** Histogram visualizing average text lengths.  
- **Word Cloud:** Displays most frequent words across messages.  
- **Class Imbalance:** Labels 0 and 1 (sadness, joy) are overrepresented compared to minority classes like love, fear, and surprise.

---

## Data Cleaning & Preprocessing
Two main approaches were used with the **SpaCy** NLP pipeline:

1. **Approach 1 – Lemmatization:**  
   - Lowercasing, removing stop words and punctuation  
   - Lemmatizing tokens to their root form  
   - Example: “crying”, “cries”, “cried” → “cry”  

2. **Approach 2 – Tokenization (No Lemmatization):**  
   - Lowercasing, removing stop words and punctuation  
   - Preserved verb tenses and plural forms  
   - Retains original emotion nuance

Both approaches were compared later for performance impact.

---

## Word Embedding Techniques
Three text-representation methods were tested on both preprocessing approaches:

### One-Hot Encoding
- Implemented via `CountVectorizer(binary=True)`  
- Represents words by binary presence (1) or absence (0)  
- Simple, effective for small datasets  

### TF-IDF
- Implemented via `TfidfVectorizer()`  
- Weighs words based on importance relative to document frequency  
- Captures distinct but rare emotional terms  

### Word2Vec
- Implemented using **Gensim** library  
- Trained both **CBOW** and **Skip-Gram** models  
- Vector size = 100, window = 5, min_count = 1  
- Compared embeddings for tokenized vs lemmatized text  

---

## Modeling & Evaluation
- **Model Used:** Logistic Regression (max_iter=1000)  
- **Validation:** 5-fold Cross-Validation for One-Hot and TF-IDF  
- **Metrics:** Accuracy, Precision, Recall, F1-Score (for Word2Vec)  

### Results Summary

| Embedding Type | Preprocessing | Avg Accuracy |
|----------------|----------------|---------------|
| TF-IDF | Lemmatized | 61.35% |
| TF-IDF | Tokenized | 61.95% |
| One-Hot | Lemmatized | 69.65% |
| One-Hot | Tokenized | **72.40%** |
| Word2Vec | Lemmatized | 34.27% |
| Word2Vec | Tokenized | 34.78% |

### Key Observations
- One-Hot Encoding with tokenized data achieved the highest accuracy (72.4%).  
- Lemmatization did not improve results — emotional nuances are lost when reducing words to roots.  
- TF-IDF performed moderately well but lacked context sensitivity.  
- Word2Vec embeddings underperformed due to short text and class imbalance.

---

## Visualization Results
- **Bar Chart:** Compared accuracies across all embedding techniques.  
- **Word Clouds:** Highlighted frequently occurring emotional terms.  
- **Label Distribution:** Visualized imbalance between dominant and minority emotion classes.

---

## Conclusion
Simpler, count-based methods such as **One-Hot Encoding** outperformed more sophisticated embeddings for this small, imbalanced dataset.  
Preserving original word forms (tokenization only) proved more effective than aggressive lemmatization.  
Word2Vec’s semantic representations were diluted in short, context-poor texts.  
Class imbalance also limited minority emotion detection performance.

---

## How to Run
```bash
# Clone the repository
git clone https://github.com/MuhammadOsama380/Emotion-Detection-in-Text-Messages.git
cd Emotion-Detection-in-Text-Messages

# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook notebooks/Osama_1288056.ipynb
```

---

## Tools & Libraries
- Python 3.x  
- Pandas  
- NumPy  
- Matplotlib / Seaborn  
- SpaCy  
- Scikit-learn  
- WordCloud  
- Gensim  

---

## Repository Structure
```
Emotion-Detection-in-Text-Messages/
│
├── data/
│   └── Text_Messages.csv
│
├── notebooks/
│   └── Osama_1288056.ipynb
│
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

## License
This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements
Developed by **Muhammad Osama (1288056)** as part of the **Natural Language Processing** coursework at **Fanshawe College**.  
If you found this project helpful, please ⭐ the repository or connect on [LinkedIn](https://www.linkedin.com/in/muhammad-osama-872328202).
