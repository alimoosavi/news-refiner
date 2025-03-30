import re
import unicodedata

import tiktoken
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Optional
import numpy as np


def get_keywords(news_list, top_n=5):
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.85, min_df=2)
    tfidf_matrix = vectorizer.fit_transform(news_list)
    feature_names = np.array(vectorizer.get_feature_names_out())
    keywords = []
    for i in range(tfidf_matrix.shape[0]):
        row = tfidf_matrix[i].toarray()[0]
        top_indices = np.argsort(row)[::-1][:top_n]
        top_keywords = feature_names[top_indices]
        keywords.append(top_keywords)
    return keywords


def count_tokens(text: str) -> int:
    encoder = tiktoken.get_encoding("cl100k_base")
    return len(encoder.encode(text))


def clean_persian_text(text):
    # Remove/replace problematic Unicode control characters (keep \n)
    text = re.sub(r'[\xad\u200c\u200d\u200e\u200f]', ' ', text)  # Replace with space

    # Remove non-Persian/non-standard characters (keep newlines, Persian letters, punctuation)
    text = re.sub(r'[^\u0600-\u06FF\s\.،؛؟!٪\d\n]', ' ', text)
    return text


def normalize_text(text):
    # Normalize Unicode (NFC form) and replace Arabic characters with Persian equivalents
    text = unicodedata.normalize('NFC', text)
    replacements = {'ك': 'ک', 'ي': 'ی', 'ة': 'ه', 'ٔ': ''}
    for arabic_char, persian_char in replacements.items():
        text = text.replace(arabic_char, persian_char)
    return text


def fix_whitespace(text):
    # Collapse multiple spaces into one (preserve newlines)
    text = re.sub(r' +', ' ', text)  # Only target spaces, not \n

    # Fix spacing around punctuation (avoid touching newlines)
    text = re.sub(r' +([،؛؟.!])', r'\1', text)  # Remove spaces BEFORE punctuation
    text = re.sub(r'([،؛؟.!]) +', r'\1 ', text)  # Add space AFTER punctuation

    return text.strip()


def preprocess_persian_document(text):
    text = clean_persian_text(text)
    text = normalize_text(text)
    text = fix_whitespace(text)
    return text


def is_irna_news_url(url: str) -> bool:
    """Check if the given URL follows the pattern 'http://www.irna.ir/<random_string>'."""
    pattern = r"^http://www\.irna\.ir/[a-zA-Z0-9]+$"
    return bool(re.match(pattern, url))


def is_isna_news_url(url: str) -> bool:
    """Check if the given URL follows the pattern 'http(s)://(www.)?isna.ir/<random_string>'."""
    pattern = r"^https?://(www\.)?isna\.ir/[a-zA-Z0-9]+$"
    return bool(re.match(pattern, url))


def extract_link(content, source=None):
    """
    Detects a link in the content, whether it starts with http, https, or without a protocol (e.g., irna.ir/xjSPkL).
    Excludes links enclosed in parentheses.
    Returns a list of detected links.
    """
    # Regex pattern to match URLs:
    # - URLs with "http://" or "https://"
    # - URLs without a protocol (e.g., "irna.ir/xjSPkL")
    # - Ensures that links do not end with parentheses
    url_pattern = r'\b(?:https?://)?(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?:/[^\s()]*)?'

    # Find all matching links
    links = re.findall(url_pattern, content)

    if source == 'IRNA':
        return [link for link in links if is_irna_news_url(link)]
    elif source == 'ISNA':
        return [link for link in links if is_isna_news_url(link)]

    return []


def extract_title_and_body(source: str, news_content: str):
    if source in {'JAHAN_FOURI', 'ISNA'}:
        match = re.search(r'\*\*(.*?)\*\*', news_content)
        if match:
            title = match.group(1)  # Extracted title without the **
            body = news_content.replace(match.group(0), '').strip()  # Remove title from content
            return title, body
    return None  # Return None if no title is found
