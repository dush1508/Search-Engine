import tkinter as tk
import webbrowser
import time
from tkinter import scrolledtext
from index_constructor import (read_index_from_file, load_json_data, read_meta_data_index_from_file,
                               read_bigram_positions)
from advanced_query import advanced_query

bookkeeping_input = "webpages_raw/bookkeeping.json"
directory_path = "webpages_raw/"
# bookkeeping_input = "test/test.json"
# directory_path = "test/"
output_file = "inverted_index.txt"
output_file_bigram = "inverted_bigram_index.txt"
meta_data_file = "meta_data_file.txt"
bigram_positions_file = 'bigram_position.txt'


class SearchEngineGUI:
    def __init__(self, master, bookkeeping_input, output_file, output_file_bigram, meta_data_file, bigram_positions_file):
        self.master = master
        master.title("Search Engine")
        master.geometry("1000x750")

        self.label = tk.Label(master, text="Enter your query:")
        self.label.pack()

        self.entry = tk.Entry(master, width=100)
        self.entry.pack()

        self.search_button = tk.Button(master, text="Search", command=self.search)
        self.search_button.pack()

        self.results_frame = tk.Frame(master)
        self.results_frame.pack(fill=tk.BOTH, expand=True)
        self.results_text = scrolledtext.ScrolledText(self.results_frame, wrap=tk.WORD, height=30)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        self.results_text.config(state=tk.DISABLED)

        start_time = time.time()
        self.index = read_index_from_file(output_file)
        self.bigram_index = read_index_from_file(output_file_bigram)
        self.json_data = load_json_data(bookkeeping_input)
        self.meta_index = read_meta_data_index_from_file(meta_data_file)
        self.bigram_positions = read_bigram_positions(bigram_positions_file)
        end_time = time.time() 
        elapsed_time = end_time - start_time
        print(f"Time for Loading Index: {elapsed_time} seconds")


    def search(self):
        start_time = time.time()
        self.results_text.config(state=tk.NORMAL)  # Set the state to NORMAL before each search
        query = self.entry.get().lower()
        result_list = advanced_query(query, self.index, self.bigram_index, self.bigram_positions, bookkeeping_input)
        self.display_results(result_list)
        self.results_text.config(state=tk.DISABLED)  # Set the state back to DISABLED after displaying the results
        end_time = time.time() 
        elapsed_time = end_time - start_time
        print(f"Search time: {elapsed_time} seconds")

    
    def display_results(self, doc_id_set):
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        if not doc_id_set:
            self.results_text.insert(tk.END, "No results found.")
        else:
            self.results_text.insert(tk.END, f"Total number of URLs found: {len(doc_id_set)}\n\n")
            for i, doc_id in enumerate(sorted(doc_id_set)[:20], start=1):
                if doc_id in self.json_data:
                    metadata = self.meta_index[doc_id]
                    title = metadata[0]
                    description = metadata[1] if len(metadata) > 1 else "Description not available"
                    link_url = "https://" + self.json_data[doc_id]
                    link_tag = f"link_{i}"
                    self.results_text.insert(tk.END, f"{i}: {title}\n", link_tag)
                    self.results_text.insert(tk.END, f"{description}\n")
                    self.results_text.tag_config(link_tag, foreground="blue", underline=True)
                    self.results_text.tag_bind(link_tag, "<Button-1>", lambda e, url=link_url: self.open_link(url))


    def open_link(self, url):
        webbrowser.open_new(url)
        