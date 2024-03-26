from nltk.corpus import stopwords
from autocorrect import Speller

def auto_correct(words):
    spell = Speller()
    corrected_words_list = [spell(word) for word in words.split()]
    corrected_phrase = ' '.join(corrected_words_list)
    return corrected_phrase


def preprocess_query(word):
    stop_words = set(stopwords.words('english'))
    word = auto_correct(word)
    word_list = word.split()
    # Filter out stopwords and create tuples of words and their positions
    processed_query = [(word, index) for index, word in enumerate(word_list) if word.lower() not in stop_words]
    return processed_query


def advanced_query(word, index, bigram_index, bigram_positions, bookkeeping_input):
    processed_query = preprocess_query(word)
    query_length = len(processed_query)
    if query_length < 1:
        result_list = []
        print("No Results Found! Try again.")
    elif query_length == 1:
        result_list = one_word_query(processed_query, index, bookkeeping_input)
    elif query_length == 2:
        result_list = two_word_query(processed_query, bigram_index, bookkeeping_input)
    else:
        result_list = multi_word_query(processed_query, bigram_index, bigram_positions, bookkeeping_input)
    return result_list


def ranked_retrieval(word, index):
    ranked_list = []
    if word in index:
        posting_list = index[word]
        for posting in posting_list:
            ranked_score = (0.5 * posting.tf_idf_score) + posting.html_tag_weight + posting.pagerank_weight
            posting.final_weight = ranked_score
            ranked_list.append(posting)

    unique_results = {}
    for item in ranked_list:
        doc_id = item.doc_id
        if doc_id not in unique_results:
            unique_results[doc_id] = item
    unique_results_list = list(unique_results.values())
    
    ranked_list = sorted(unique_results_list, key=lambda x: x.final_weight, reverse=True)
    return ranked_list


def get_results(ranked_list, bookkeeping_input):
    """
    Displays the results based on the document IDs found in the inverted index.
    """
    result_list = []
    if not ranked_list:
        return result_list
    else:
        for posting in ranked_list:
            result_list.append(posting.doc_id)
    return  result_list


def one_word_query(processed_query, index, bookkeeping_input):
    """
    Performs a one word query and displays the results.
    """
    word = processed_query[0][0]
    ranked_list = ranked_retrieval(word, index)
    result_list = get_results(ranked_list, bookkeeping_input)
    return result_list


def two_word_query(processed_query, bigram_index, bookkeeping_input):
    """
    Performs a two word query and displays the results.
    """
    word = processed_query[0][0] + " " + processed_query[1][0]
    ranked_list = ranked_retrieval(word, bigram_index)
    result_list = get_results(ranked_list, bookkeeping_input)
    return result_list


def multi_word_query(processed_query, bigram_index, bigram_positions, bookkeeping_input):
    """
    Performs a query and displays the results.
    """
    results_list = []
    bigram_pairs = generate_bigram_pairs(processed_query, bigram_index, results_list)
    ranked_list = multi_word_ranked_retrieval(bigram_pairs, results_list, bigram_index, bigram_positions)
    result_list = get_results(ranked_list, bookkeeping_input)
    return result_list


def multi_word_ranked_retrieval(bigram_pairs, results_list, bigram_index, bigram_positions):
    add_word_position_score(bigram_pairs, results_list, bigram_index, bigram_positions)
    add_normalized_vector_weight(bigram_pairs, results_list, bigram_index)
    for posting in results_list:
        ranked_score = ((0.5 * posting.tf_idf_score) + posting.html_tag_weight + posting.pagerank_weight
                        + (4 * posting.normalized_vector_weight))
        posting.final_weight = ranked_score

    unique_results = {}
    for item in results_list:
        doc_id = item.doc_id
        if doc_id not in unique_results:
            unique_results[doc_id] = item
    unique_results_list = list(unique_results.values())
    
    ranked_list = sorted(unique_results_list, key=lambda x: x.final_weight, reverse=True)

    return ranked_list


def generate_bigram_pairs(processed_query, bigram_index, results_list):
    bigram_pairs = []
    for i in range(len(processed_query) - 1):
        bigram = processed_query[i][0] + " " + processed_query[i + 1][0]
        difference = processed_query[i + 1][1] - processed_query[i][1]
        bigram_pairs.append((bigram, difference))
        if bigram in bigram_index:
            results_list.extend(bigram_index[bigram])
    return bigram_pairs


def add_word_position_score(bigram_pairs, results_list, bigram_index, bigram_positions):
    for i in range(len(bigram_pairs) - 1):
        bigram1, bigram1_pos = bigram_pairs[i]
        bigram2, bigram2_pos = bigram_pairs[i + 1]
        difference = bigram2_pos - bigram1_pos
        proximity_range = 5 + difference
        
        if bigram1 not in bigram_index or bigram2 not in bigram_index:
            continue

        postings1 = bigram_index[bigram1]
        postings2 = bigram_index[bigram2]

        doc_id_set1 = set(posting.doc_id for posting in postings1)
        doc_id_set2 = set(posting.doc_id for posting in postings2)
        common_doc_ids = doc_id_set1.intersection(doc_id_set2)

        for doc_id in common_doc_ids:
            bigram_tup1 = (bigram1, doc_id)
            bigram_tup2 = (bigram2, doc_id)

            if bigram_tup1 not in bigram_positions or bigram_tup2 not in bigram_positions:
                continue

            positions1 = bigram_positions[bigram_tup1]
            positions2 = bigram_positions[bigram_tup2]

            for pos1 in positions1:
                for pos2 in positions2:
                    distance = abs(pos1 - pos2)

                    if distance <= proximity_range:
                        for k in range(len(results_list)):
                            if results_list[k].doc_id == doc_id:
                                results_list[k].final_weight += 1
                                round(results_list[k].final_weight, 2)
                    elif distance > proximity_range:
                        for k in range(len(results_list)):
                            if results_list[k].doc_id == doc_id:
                                results_list[k].final_weight += 0.1
                                round(results_list[k].final_weight, 2)


def add_normalized_vector_weight(bigram_pairs, results_list, bigram_index):
    normalized_denominator = 0
    for i, (bigram, value) in enumerate(bigram_pairs):
        if bigram in bigram_index:
            idf = bigram_index[bigram][0].idf
            bigram_pairs[i] = (bigram, idf)
    for bigram, idf in bigram_pairs:
        normalized_denominator += idf ** 2
    normalized_denominator = normalized_denominator ** 0.5
    for posting in results_list:
        normalized_vector_weight = round(posting.normalized_vector_weight / normalized_denominator, 3)
        posting.normalized_vector_weight = normalized_vector_weight