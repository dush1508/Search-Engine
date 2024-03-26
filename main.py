import time
import tkinter as tk
from index_constructor import build_inverted_index, write_bigram_positions
from gui import SearchEngineGUI

bookkeeping_input = "webpages_raw/bookkeeping.json"
directory_path = "webpages_raw/"
# bookkeeping_input = "test/test.json"
# directory_path = "test/"
output_file = "inverted_index.txt"
output_file_bigram = "inverted_bigram_index.txt"
meta_data_file = "meta_data_file.txt"
bigram_positions_file = 'bigram_position.txt'


def build_index():
    print("Building inverted index...")
    start_time = time.time()
    build_inverted_index(bookkeeping_input, directory_path, output_file, output_file_bigram, meta_data_file)
    write_bigram_positions(bookkeeping_input, directory_path, bigram_positions_file)
    end_time = time.time()
    elapsed_time_minutes = round((end_time - start_time) / 60, 4)
    print(f"Elapsed time for building the inverted index: {elapsed_time_minutes} minutes")


def main():
    root = tk.Tk()
    # build_index()
    gui = SearchEngineGUI(root, bookkeeping_input, output_file, output_file_bigram, meta_data_file, bigram_positions_file)
    root.mainloop()


if __name__ == "__main__":
    main()
