import os
import json
import numpy as np
import math
import re
from collections import Counter


def load_data(json_file_path):
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
    return data


def load_documents(candidate_documents):
    documents = []
    for i in candidate_documents:
        filename = f"document_{i}.txt"
        file_path = os.path.join("data", filename)

        with open(file_path, "r", encoding="utf-8") as file:
            document = file.read()
            documents.append(document)

    return documents


def tf(term, document):
    # Calculate Term Frequency (TF)
    term_frequency = document.count(term)
    return term_frequency


def idf(term, all_documents):
    # Calculate Inverse Document Frequency (IDF)
    document_frequency = sum([1 for doc in all_documents if term in doc])
    if document_frequency == 0:
        return 0
    else:
        return math.log(len(all_documents) / document_frequency)


def tf_idf(term, document, all_documents):
    # Calculate TF-IDF
    term_frequency = tf(term, document)
    inverse_document_frequency = idf(term, all_documents)

    return term_frequency * inverse_document_frequency


def cos_similarity(vector1, vector2):
    # Calculate cosine similarity between two vectors
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)

    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity


def most_common_words(document):
    # Count the occurrences of each term in the document
    term_counts = Counter(document.split())

    # Find the three most common terms
    most_common_terms = term_counts.most_common(3)

    return most_common_terms


def split_into_paragraphs(document):
    # Split document into paragraphs based on empty lines
    paragraphs = document[0].split('\n')
    return paragraphs


def main():
    json_file_path = 'data.json'
    data = load_data(json_file_path)
    for i in range(101):
        query_data = data[i]
        query = query_data['query']
        candidate_documents = query_data['candidate_documents_id']

        documents = load_documents(candidate_documents)

        # Get all unique terms from both the query and documents
        all_terms = set(query.split())
        for document in documents:
            all_terms.update(document.split())

        # Vectorize the query
        query_vector = [tf_idf(term, query, documents) for term in all_terms]

        # Create a list to store cosine similarity values for each document
        similarity_values = []
        para_similarity_values = []
        common_words = {}
        top_5_terms = []
        j = 0
        k = candidate_documents[j]
        # Iterate over each document
        for document in documents:
            j = j + 1
            # Vectorize the document
            doc_vector = [tf_idf(term, document, documents)
                          for term in all_terms]
            term_tfidf_dict = {term: tf_idf(
                term, document, documents) for term in all_terms}

            # Calculate cosine similarity between the query and the document
            similarity = cos_similarity(query_vector, doc_vector)
            similarity_values.append(similarity)

        # Find the index of the document with the highest cosine similarity
        most_similar_document_index = np.argmax(similarity_values)

        paragraphs = load_documents(
            [candidate_documents[most_similar_document_index]])
        paragraphs_sep = split_into_paragraphs(paragraphs)

        for para in paragraphs_sep:
            # Vectorize the document
            para_vector = [tf_idf(term, para, paragraphs_sep)
                           for term in all_terms]

            # Calculate cosine similarity between the query and the document
            para_similarity = cos_similarity(query_vector, para_vector)
            para_similarity_values.append(para_similarity)

        # Find the index of the document with the highest cosine similarity
        most_similar_document_index = np.argmax(similarity_values)
        most_similar_para_index = np.argmax(para_similarity_values)
        # Print the most similar document and its cosine similarity
        print(f"The most similar document to query {i} is document {candidate_documents[most_similar_document_index]} "
              f"with a cosine similarity of {similarity_values[most_similar_document_index]}.\n")
        print(f"most similar paragraph:{most_similar_para_index}")


if __name__ == "__main__":
    main()
