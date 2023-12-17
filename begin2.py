import os
import json
import numpy as np
import math


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


def calculate_cosine_similarity(query, documents):
    def tfidf_vectorizer(corpus):
        word_set = set()
        tfidf_vectors = []

        for document in corpus:
            words = document.lower().split()
            word_set.update(words)

        word_list = list(word_set)
        word_list.sort()

        for document in corpus:
            words = document.lower().split()
            tf_vector = [words.count(word) for word in word_list]
            tfidf_vectors.append(tf_vector)

        idf_vector = [math.log(len(corpus) / (1 + sum([1 for doc in corpus if word in doc.lower().split()])))
                      for word in word_list]

        for i in range(len(tfidf_vectors)):
            for j in range(len(idf_vector)):
                tfidf_vectors[i][j] *= idf_vector[j]

        return tfidf_vectors

    def cosine_similarity(vec1, vec2):
        dot_product = sum([a * b for a, b in zip(vec1, vec2)])
        norm_vec1 = math.sqrt(sum([a ** 2 for a in vec1]))
        norm_vec2 = math.sqrt(sum([b ** 2 for b in vec2]))

        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0
        else:
            return dot_product / (norm_vec1 * norm_vec2)

    def split_into_lines(document):
        return document.split('\n')

    def calculate_line_similarity(query, lines):
        line_similarities = [cosine_similarity(
            tfidf_vectorizer([query, line])[0], tfidf_vectorizer([line, line])[1]) for line in lines]
        return line_similarities

    tfidf_matrix = tfidf_vectorizer([query] + documents)
    similarities = [cosine_similarity(
        tfidf_matrix[0], tfidf_matrix[i]) for i in range(1, len(tfidf_matrix))]
    most_similar_index = np.argmax(similarities)

    most_similar_document = documents[most_similar_index]
    most_similar_document_lines = split_into_lines(most_similar_document)
    line_similarities = calculate_line_similarity(
        query, most_similar_document_lines)

    most_similar_line_index = np.argmax(line_similarities)
    lenght = len(split_into_lines(most_similar_document))
    return most_similar_index, most_similar_line_index, lenght


def main():
    json_file_path = 'data.json'
    data = load_data(json_file_path)
    for i in range(11):
        query_data = data[i]
        query = query_data['query']
        candidate_documents = query_data['candidate_documents_id']

        documents = load_documents(candidate_documents)

        most_similar_document, most_similar_line, lenght = calculate_cosine_similarity(
            query, documents)
        list1 = []
        for k in range(0, lenght+1):
            if k == most_similar_line:
                list1.append("1")
            else:
                list1.append("0")
        print(
            f"Is selected:{list1}"
        )

        print(
            f"The most similar document to the query{i} is:{candidate_documents[most_similar_document]}\n"
        )


if __name__ == "__main__":
    main()
