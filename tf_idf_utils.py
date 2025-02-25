import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from hazm import stopwords_list


def clean_persian_text(text):
    """ Clean Persian text (remove unwanted characters, punctuation, etc.). """
    # Normalize text
    text = re.sub(r'[\u200c\u200b]', '', text)  # Remove zero-width spaces
    text = re.sub(r'[\n\r\t]', ' ', text)  # Remove newlines, tabs
    text = re.sub(r'[^a-zA-Z0-9\u0600-\u06FF\s]', '', text)  # Keep Persian characters and numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text


def get_tfidf_vectors(news_list, query):
    # Clean the Persian text
    cleaned_news = [clean_persian_text(news) for news in news_list]
    cleaned_query = clean_persian_text(query)  # Clean the query

    # Create the TF-IDF vectorizer with Persian stop words
    vectorizer = TfidfVectorizer(stop_words=stopwords_list(),
                                 max_df=0.95,  # Increased max_df
                                 min_df=1)  # Lowered min_df

    # Fit and transform the TF-IDF matrix for the news articles
    tfidf_matrix = vectorizer.fit_transform(cleaned_news)

    # Transform the query using the same vectorizer
    query_vector = vectorizer.transform([cleaned_query])

    return tfidf_matrix, query_vector


def similarity_search(news_list, query):
    # Get the TF-IDF matrix for the news articles and query
    tfidf_matrix, query_vector = get_tfidf_vectors(news_list, query)

    # Compute cosine similarity between the query and each news article
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Get the indices of the news articles sorted by similarity (descending order)
    sorted_indices = np.argsort(cosine_similarities)[::-1]

    # Print the similarities along with the news articles
    for i in sorted_indices:
        print(f"Similarity: {cosine_similarities[i]:.4f} - {news_list[i]}")


# Example news articles
news_list = [
    "دولت ایران در حال توسعه سیاست‌های جدید اقتصادی است.",
    "بازار بورس تهران با افت قیمت‌های سهام روبرو شده است.",
    "تحلیل‌گران پیش‌بینی می‌کنند که قیمت طلا در ماه آینده افزایش یابد."
]

# Sample query
query = "دولت در خصوص اقتصاد در حال چه کاری است؟"

if __name__ == "__main__":
    similarity_search(news_list, query)
