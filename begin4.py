import os
import json
import numpy as np
import math
from collections import Counter
import string


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
    term_frequency = document.count(term)
    return term_frequency


def idf(term, all_documents):
    document_frequency = sum([1 for doc in all_documents if term in doc])
    if document_frequency == 0:
        return 0
    else:
        return math.log(len(all_documents) / document_frequency)


def tf_idf(term, document, all_documents):
    term_frequency = tf(term, document)
    inverse_document_frequency = idf(term, all_documents)

    return term_frequency * inverse_document_frequency


def cos_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)

    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity


def split_into_paragraphs(document):
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

        all_terms = set(query.split())
        for document in documents:
            all_terms.update(document.split())

        query_vector = [tf_idf(term, query, documents) for term in all_terms]

        similarity_values = []
        para_similarity_values = []
        top_5_terms = []
        j = 0
        k = candidate_documents[j]
        for document in documents:
            j = j + 1
            document = document.translate(
                str.maketrans("", "", string.punctuation))
            doc_vector = [tf_idf(term, document, documents)
                          for term in all_terms]

            sorted_tfidf = sorted(enumerate(doc_vector),
                                  key=lambda x: x[1], reverse=True)

            all_terms_list = list(all_terms)
            top_5_terms = [all_terms_list[index]
                           for index, _ in sorted_tfidf[:5]]

            term_counts = {term: tf(term, document)
                           for term in all_terms if len(term) > 1}

            most_repeated_words = Counter(term_counts).most_common(5)
            similarity = cos_similarity(query_vector, doc_vector)
            similarity_values.append(similarity)

        most_similar_document_index = np.argmax(similarity_values)

        paragraphs = load_documents(
            [candidate_documents[most_similar_document_index]])
        paragraphs_sep = split_into_paragraphs(paragraphs)

        for para in paragraphs_sep:
            para_vector = [tf_idf(term, para, paragraphs_sep)
                           for term in all_terms]

            para_similarity = cos_similarity(query_vector, para_vector)
            para_similarity_values.append(para_similarity)

        most_similar_document_index = np.argmax(similarity_values)
        most_similar_para_index = np.argmax(para_similarity_values)
        print(f"The most similar document to query {i} is document {candidate_documents[most_similar_document_index]} "
              f"with a cosine similarity of {similarity_values[most_similar_document_index]}.\n")
        print(f"most similar paragraph:{most_similar_para_index}")


if __name__ == "__main__":
    main()
