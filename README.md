Here's a README for your project:

---

# Amazon Product Recommendation System

This project implements a content-based recommendation system for Amazon products using the **TF-IDF Vectorizer** and **cosine similarity** to recommend similar products based on their titles, descriptions, and bullet points. It also supports partial title matching using the **fuzzywuzzy** library for a better user experience.

---

## Features

- **Content-Based Recommendations**: Finds similar products based on combined features like product title, bullet points, and description.
- **TF-IDF Vectorizer**: Extracts important words from product descriptions and creates a feature matrix for similarity computation.
- **Cosine Similarity**: Measures similarity between product vectors to recommend the most relevant products.
- **Partial Title Matching**: Handles cases where the user input is incomplete or approximate, improving flexibility and usability.

---

## Dataset

- **Dataset**: [Amazon Product Data](https://www.kaggle.com/datasets/piyushjain16/amazon-product-data)
- **Format**: Contains columns such as `TITLE`, `BULLET_POINTS`, `DESCRIPTION`, and more.
- **Preprocessing**:
  - Rows with missing `TITLE` values are dropped.
  - Missing `BULLET_POINTS` and `DESCRIPTION` are filled with default values.
  - Text is cleaned by removing punctuation, converting to lowercase, and removing stopwords.

---

## Setup Instructions

### Prerequisites
- Python 3.x
- Libraries: `pandas`, `nltk`, `sklearn`, `fuzzywuzzy`

### Install Dependencies
```bash
pip install pandas scikit-learn fuzzywuzzy[speedup] nltk
```

### Download Dataset
1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/piyushjain16/amazon-product-data).
2. Unzip and place the dataset in the project directory.

---

## How to Run

1. **Prepare Dataset**:
    ```python
    import pandas as pd
    import re
    import nltk
    from nltk.corpus import stopwords

    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    df = pd.read_csv('/content/dataset/train.csv')
    df = df.dropna(subset=['TITLE'])
    df['BULLET_POINTS'] = df['BULLET_POINTS'].fillna('No bullet points available')
    df['DESCRIPTION'] = df['DESCRIPTION'].fillna('No description available')
    df['combined_features'] = (df['TITLE'] + " " + df['BULLET_POINTS'] + " " + df['DESCRIPTION'])

    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = " ".join([word for word in text.split() if word not in stop_words])
        return text

    df['combined_features'] = df['combined_features'].apply(clean_text)
    ```

2. **Generate TF-IDF Matrix**:
    ```python
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['combined_features'])
    ```

3. **Recommendation Function**:
    ```python
    from sklearn.metrics.pairwise import linear_kernel
    from fuzzywuzzy import process

    def recommend_product_by_title_sparse(product_title, top_n=5):
        matched_title, confidence = process.extractOne(product_title, df['TITLE'].values)
        if confidence < 70:
            return f"No close match found for '{product_title}' in the dataset."

        print(f"Matched Product Title: '{matched_title}' (Confidence: {confidence}%)")
        product_idx = df[df['TITLE'] == matched_title].index[0]
        cosine_similarities = linear_kernel(tfidf_matrix[product_idx], tfidf_matrix).flatten()
        similar_indices = cosine_similarities.argsort()[::-1][1:top_n+1]
        return df.iloc[similar_indices]['TITLE'].tolist()
    ```

4. **Test the Function**:
    ```python
    input_title = "Marks & Spencer Girls' Pyjama"
    recommendations = recommend_product_by_title_sparse(input_title, top_n=5)
    print("Recommended Products:")
    for product in recommendations:
        print(product)
    ```

---

## Example Outputs

- **Input**: `"Marks & Spencer Girls' Pyjama"`
    - **Recommended Products**:
        1. Product Title 1
        2. Product Title 2
        3. Product Title 3
        4. Product Title 4
        5. Product Title 5

- **Input**: `"Nike Women's"`
    - **Recommended Products**:
        1. Product Title A
        2. Product Title B
        3. Product Title C
        4. Product Title D
        5. Product Title E

---

## Future Improvements
- Use advanced embeddings like Word2Vec, FastText, or BERT for more robust recommendations.
- Add user-based or hybrid collaborative filtering to enhance recommendations.

---

## License
This project is licensed under the [MIT License](LICENSE).

--- 

Let me know if you'd like to further customize this README!
