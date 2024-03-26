class PostingObject:
    def __init__(self, doc_id="", frequency=1, idf=0, tf_idf_score=0, html_tag_weight=0, pagerank_weight=0,
                 normalized_vector_weight=0, final_weight=0):
        self.doc_id = doc_id
        self.frequency = frequency
        self.idf = idf
        self.tf_idf_score = tf_idf_score
        self.normalized_vector_weight = normalized_vector_weight
        self.html_tag_weight = html_tag_weight
        self.pagerank_weight = pagerank_weight
        self.final_weight = final_weight

    def __str__(self):
        return (f"Posting Object: [Doc ID: {self.doc_id}, Frequency: {self.frequency}, "
                f"TF-IDF Score: {self.tf_idf_score}, IDF: {self.idf}, "
                f"HTML Tag Weight: {self.html_tag_weight}, Pagerank Weight: {self.pagerank_weight}, "
                f"Normalized Vector Weight: {self.normalized_vector_weight}, "
                f"Final Weight: {self.final_weight}]")