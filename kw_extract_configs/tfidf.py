# from itertools import islice
# from tqdm.notebook import tqdm
# from re import sub

# dict_idf = {}
# with tqdm(total=num_lines) as pbar:
#     for i, line in tqdm(islice(enumerate(file), 1, None)):
#         try: 
#             cells = line.split()
#             idf = float(sub("[^0-9.]", "", cells[3]))
#             dict_idf[cells[0]] = idf
#         except: 
#             print("Error on: " + line)
#         finally:
#             pbar.update(1)



# from sklearn.feature_extraction.text import CountVectorizer
# from numpy import array, log

# vectorizer = CountVectorizer()
# tf = vectorizer.fit_transform([x.lower() for x in array_text])
# tf = tf.toarray()
# tf = log(tf + 1)


# tfidf = tf.copy()
# words = array(vectorizer.get_feature_names())
# for k in tqdm(dict_idf.keys()):
#     if k in words:
#         tfidf[:, words == k] = tfidf[:, words == k] * dict_idf[k]
#     pbar.update(1)

# for j in range(tfidf.shape[0]):
# print("Keywords of article", str(j+1), words[tfidf[j, :].argsort()[-5:][::-1]])



class TfIdf:
    def __init__(self):
        self.weighted = False
        self.documents = []
        self.corpus_dict = {}

    def add_document(self, doc_name, list_of_words):
        # building a dictionary
        doc_dict = {}
        for w in list_of_words:
            doc_dict[w] = doc_dict.get(w, 0.) + 1.0
            self.corpus_dict[w] = self.corpus_dict.get(w, 0.0) + 1.0

        # normalizing the dictionary
        length = float(len(list_of_words))
        for k in doc_dict:
            doc_dict[k] = doc_dict[k] / length

        # add the normalized document to the corpus
        self.documents.append([doc_name, doc_dict])

    def similarities(self, list_of_words):
        """
        Returns a list of all the [docname, similarity_score] pairs relative to a list of words.
        """

        # building the query dictionary
        query_dict = {}
        for w in list_of_words:
            query_dict[w] = query_dict.get(w, 0.0) + 1.0

        # normalizing the query
        length = float(len(list_of_words))
        for k in query_dict:
            query_dict[k] = query_dict[k] / length

        # computing the list of similarities
        sims = []
        for doc in self.documents:
            score = 0.0
            doc_dict = doc[1]
            for k in query_dict:
                if k in doc_dict:
                    score += (query_dict[k] / self.corpus_dict[k]) + (
                      doc_dict[k] / self.corpus_dict[k])
            sims.append([doc[0], score])

        return sims
