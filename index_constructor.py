import json
import nltk
import re
import math
import networkx as nx
from bs4 import BeautifulSoup
from nltk.corpus import stopwords, words
from nltk.stem import WordNetLemmatizer
from PostingObject import PostingObject

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('words')


def preprocess_text(content):
    """
    Tokenize the content using NLTK, remove stopwords, perform lemmatization,
    and filter out terms not present in NLTK's word corpus.
    """
    tokens = re.split(r'[^a-zA-Z]', content.lower())
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = []
    english_words = set(words.words())
    # Lemmatize known words, otherwise add them as they are
    for token in tokens:
        if token not in stop_words and len(token) > 1:
            if token in english_words:
                lemmatized_tokens.append(lemmatizer.lemmatize(token))
            else:
                lemmatized_tokens.append(token)
    return lemmatized_tokens


def load_json_data(bookkeeping_input):
    """
    Load JSON data from the specified file path.
    """
    with open(bookkeeping_input, "r", encoding="utf-8") as json_file:
        return json.load(json_file)


def parse_html_content(content):
    # Initialize lists to store extracted data
    meta_texts = []
    titles = []
    headings = []
    bold_texts = []
    remaining_text = []
    anchor_texts = []
    links = []

    # Parse the HTML content
    soup = BeautifulSoup(content, 'lxml')

    # Extract text content from the soup object
    text_content = soup.find_all(text=True)
    text_content = ' '.join(text_content)

    # Get text for description
    body_content = soup.body.get_text().strip() if soup.body else ""

    # Split the text content into words based on whitespace
    words = preprocess_text(text_content)

    # Iterate through the list of words and add each word with its position
    word_positions = {}
    for position, word in enumerate(words):
        if word not in word_positions:
            word_positions[word] = []
        word_positions[word].append(position)

    # Iterate through the list of words and add each bigram with its position
    bigram_positions = {}
    for position, (first, second) in enumerate(zip(words, words[1:])):
        bigram = first + " " + second
        if bigram not in bigram_positions:
            bigram_positions[bigram] = []
        bigram_positions[bigram].append(position)

    # Extract meta tags
    meta_tags = soup.find_all('meta')
    for tag in meta_tags:
        content = tag.get('content', '').strip()
        if content:
            meta_texts.append(content)
        tag.extract()

    # Extract title
    title_tag = soup.title
    if title_tag:
        title = title_tag.string.strip() if title_tag.string else ''
        title_tag.extract()
        titles.append(title)

    # Extract headings (h1 to h6)
    for heading_tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        heading_text = heading_tag.get_text(strip=True)
        headings.append(heading_text)
        heading_tag.extract()

    # Extract bold texts
    for bold_tag in soup.find_all('b'):
        bold_text = bold_tag.get_text(strip=True)
        bold_texts.append(bold_text)
        bold_tag.extract()

    # Extract links and anchor texts
    for link in soup.find_all('a', href=True):
        anchor_text = link.get_text(strip=True)
        if anchor_text:  # Check if the link text is not empty
            anchor_texts.append(anchor_text)
        link_href = link['href']
        if link_href:
            links.append(link_href)

    # Extract remaining text
    text_elements = soup.find_all(text=True)
    for element in text_elements:
        text = element.strip()
        if text:  # Exclude empty strings
            remaining_text.append(text)

    # Join the remaining text elements into a single string
    remaining_text_str = ' '.join(remaining_text)
    return (titles, headings, meta_texts, bold_texts, anchor_texts, links, remaining_text_str, body_content,
            word_positions,bigram_positions)


def build_meta_data_file(meta_data_index, file_id, title_text, description_text):
    # Check if the file_id already exists in meta_data_index
    if file_id in meta_data_index:
        # Update the existing entry with the new title_text and description_text
        meta_data_index[file_id] = (title_text, description_text)
    else:
        # Create a new entry with file_id as key and a tuple containing title_text and description_text as value
        meta_data_index[file_id] = (title_text, description_text)


def write_meta_data_index_to_file(meta_data_index, filename):
    with open(filename, 'w') as file:
        json.dump(meta_data_index, file)


def read_meta_data_index_from_file(filename):
    with open(filename, 'r') as file:
        meta_data_index = json.load(file)
    return meta_data_index


def build_inverted_index(bookkeeping_input, directory_path, output_file, output_file_2g, meta_data_file):
    """
    Build the inverted index using a dictionary from the JSON data and write it to a file.
    """
    inverted_index = {}
    inverted_bigram_index = {}
    out_links = {}
    valid_links = {}
    meta_data_index = {}
    pagerank = {}
    # word_with_doc_id = {}
    bigram_with_doc_id = {}
    # doc_id_with_length = {}
    doc_id_with_length_bigram = {}
    json_data = load_json_data(bookkeeping_input)
    for doc_id, link in json_data.items():
        valid_links[link] = doc_id
    total_docs = len(json_data)
    for doc_id, link in json_data.items():
        file_path = directory_path + doc_id
        print(file_path)
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

            (title, headings, meta_texts, bold_texts, anchor_texts, links, remaining_text_str, body_content,
             word_positions, bigram_positions) = parse_html_content(content)

            for out_link in links:
                out_links.setdefault(link, []).append(out_link)

            title_text, description_text = build_initial_index(title, headings, meta_texts, bold_texts, anchor_texts,
                                                               remaining_text_str, body_content, inverted_index,
                                                               inverted_bigram_index, doc_id)
            build_meta_data_file(meta_data_index, doc_id, title_text, description_text)
            # create_doc_id_with_word(word_with_doc_id, doc_id, word_positions)
            create_doc_id_with_word(bigram_with_doc_id, doc_id, bigram_positions)

    calculate_tf_idf(inverted_index, total_docs)
    calculate_tf_idf(inverted_bigram_index, total_docs)

    # calculate_vector_length(doc_id_with_length, final_index, word_with_doc_id)
    calculate_vector_length(doc_id_with_length_bigram, inverted_bigram_index, bigram_with_doc_id)

    # add_normalized_vector(final_index, doc_id_with_length)
    add_normalized_vector(inverted_bigram_index, doc_id_with_length_bigram)

    pagerank = calculate_pagerank(out_links, valid_links)
    add_pagerank_values(inverted_index, pagerank)
    add_pagerank_values(inverted_bigram_index, pagerank)

    write_index_to_file(inverted_index, output_file)
    write_index_to_file(inverted_bigram_index, output_file_2g)
    write_meta_data_index_to_file(meta_data_index, meta_data_file)


def create_doc_id_with_word(word_with_doc_id, file_id, word_positions):
    word_with_doc_id[file_id] = []
    for word, _ in word_positions.items():
        word_with_doc_id[file_id].append(word)


def build_initial_index(title, headings, meta_texts, bold_texts, anchor_texts, remaining_text_str, body_content,
                        initial_index, initial_bigram_index, file_id):
    """
    Build an initial version of the index.
    """
    # Preprocess text and convert to tuples to make it hashable
    title_token = tuple(preprocess_text(' '.join(title)))
    heading_token = tuple(preprocess_text(' '.join(headings)))
    meta_token = tuple(preprocess_text(' '.join(meta_texts)))
    bold_anchor_token = tuple(preprocess_text(' '.join(bold_texts + anchor_texts)))
    regular_token = tuple(preprocess_text(' '.join([remaining_text_str])))

    html_tag_value_dict = {
        title_token: 3,
        heading_token: 1.5,
        meta_token: 1.25,
        bold_anchor_token: 1,
        regular_token: 0
    }

    for token_list in [title_token, heading_token, meta_token, bold_anchor_token, regular_token]:
        for token in token_list:
            html_tag_value = html_tag_value_dict.get(token, 0)
            add_or_update_posting_list(token, initial_index, file_id, html_tag_value)

    all_text = ' '.join(title + headings + meta_texts + bold_texts + [remaining_text_str])
    tokens = preprocess_text(all_text)

    for first, second in zip(tokens, tokens[1:]):
        html_tag_value = html_tag_value_dict.get(first, 0)
        token = f"{first} {second}"
        add_or_update_posting_list(token, initial_bigram_index, file_id, html_tag_value)
    
    title_text = ' '.join(title)
    valid_description = preprocess_text(body_content)
    description_text = ' '.join(valid_description[:20])

    return title_text, description_text


def add_or_update_posting_list(token, initial_index, file_id, html_tag_value):
    found = False
    if token in initial_index:
        for posting in initial_index[token]:
            if posting.doc_id == file_id:
                posting.frequency += 1
                posting.html_tag_weight += html_tag_value
                found = True
    if not found:
        posting = PostingObject(doc_id=file_id, html_tag_weight=html_tag_value)
        initial_index.setdefault(token, []).append(posting)


def calculate_tf_idf(initial_index, total_docs):
    """
    Calculate TF-IDF for each term in the index and update the tf_idf_score attribute of each posting_object.
    """
    for _, postings_list in initial_index.items():
        n = total_docs
        df = len(postings_list)
        idf = math.log10(n / df) if df != 0 else 0
        for posting_obj in postings_list:
            tf = posting_obj.frequency
            tf = (1 + math.log10(tf)) if tf != 0 else 0
            tf_idf = tf * idf
            tf_idf_score = round(tf_idf, 3)
            idf_score = round(idf, 3)
            posting_obj.idf = idf_score
            posting_obj.tf_idf_score = tf_idf_score


def add_normalized_vector(index, doc_id_with_length):
    for _, posting_list in index.items():
        for posting in posting_list:
            if posting.doc_id in doc_id_with_length:
                if doc_id_with_length[posting.doc_id] != 0:
                    normalized_vector_weight = round(posting.tf_idf_score/doc_id_with_length[posting.doc_id], 3)
                    posting.normalized_vector_weight = normalized_vector_weight
                else:
                    posting.normalized_vector_weight = 0


def calculate_vector_length(doc_id_with_length, index, word_with_doc_id):
    for d_id, word_list in word_with_doc_id.items():
        length = 0
        if word_list:
            for word in word_list:
                if word in index:
                    posting_list = index[word]
                    for posting in posting_list:
                        if posting.doc_id == d_id:
                            length += posting.tf_idf_score ** 2
        doc_id_with_length[d_id] = length ** 0.5


def calculate_pagerank(out_links, valid_links):
    graph = nx.DiGraph()
    graph.add_nodes_from(out_links.keys())

    # Iterate over each link and its targets in out_links
    for link, targets in out_links.items():
        edges = [(link, target) for target in targets]
        graph.add_edges_from(edges)
    initial_pagerank = nx.pagerank(graph)
    pagerank = {}
    for link, value in initial_pagerank.items():
        if link in valid_links:
            pagerank[valid_links[link]] = round(value * 1000000, 3)
    
    return pagerank


def add_pagerank_values(index, pagerank):
    for _, postings_list in index.items():
        for posting in postings_list:
            if posting.doc_id in pagerank:
                posting.pagerank_weight = pagerank[posting.doc_id]


def write_index_to_file(inverted_index, filename):
    with open(filename, 'w', encoding="utf-8") as file:
        for word, postings in inverted_index.items():
            postings_str = ', '.join([
                f"{posting.doc_id}|{posting.frequency}|{posting.idf}|{posting.tf_idf_score}|"
                f"{posting.html_tag_weight}|{posting.pagerank_weight}|{posting.normalized_vector_weight}|"
                f"{posting.final_weight}"
                for posting in postings if posting is not None])
            file.write(f"{word}: {postings_str}\n")


def read_index_from_file(filename):
    inverted_index = {}
    with open(filename, 'r', encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split(': ', 1)  # Split only at the first colon
            if len(parts) == 2:
                word, postings_str = parts
                postings_list = []
                postings_data = postings_str.split(', ')
                for posting_data in postings_data:
                    data = posting_data.split('|')
                    if len(data) == 8:  # Updated for 8 fields in the posting
                        (doc_id, frequency, idf, tf_idf_score, html_tag_weight, pagerank_weight,
                         normalized_vector_weight, final_weight) = data
                        posting = PostingObject(
                            doc_id=doc_id,
                            frequency=int(frequency),
                            idf=float(idf),
                            tf_idf_score=float(tf_idf_score),
                            html_tag_weight=float(html_tag_weight),
                            pagerank_weight=float(pagerank_weight),
                            normalized_vector_weight=float(normalized_vector_weight),
                            final_weight=float(final_weight)
                        )
                        postings_list.append(posting)
                inverted_index[word] = postings_list
    return inverted_index


def write_bigram_positions(bookkeeping_input, directory_path, bigram_position_file):
    json_data = load_json_data(bookkeeping_input)
    with open(bigram_position_file, 'w') as file:
        for doc_id, _ in json_data.items():
            file_path = directory_path + doc_id
            with open(file_path, "r", encoding="utf-8") as content_file:
                content = content_file.read()
                parsed_content = parse_html_content(content)
                bigram_positions = parsed_content [9]
                for bigram, positions in bigram_positions.items():
                    sorted_positions = sorted(set(positions))
                    positions = list(sorted_positions)
                    file.write(f"({bigram}, {doc_id}): {positions}\n")


def read_bigram_positions(file_path):
    bigram_positions = {}

    with open(file_path, 'r') as file:
        for line in file:
            key_str, value_str = line.strip().split(': ')
            key = tuple(map(str.strip, key_str.strip('()').split(',')))
            value = [int(num) for num in value_str.strip('[]').split(', ')]
            bigram_positions[key] = value

    return bigram_positions