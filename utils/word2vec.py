import numpy as np
import gensim
from transformers import BertTokenizer
import gensim.models as g
import torch


class Doc2vecs():
    def __init__(self, d2v_model_path='./enwiki_dbow/doc2vec.bin', w2v_model_path="./GoogleNews-vectors-negative300.bin.gz"):
        super(Doc2vecs, self).__init__()
        self.d2v_model = g.Doc2Vec.load(d2v_model_path)
        # self.w2v_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_model_path, binary=True)
        # self.vocab = self.w2v_model.vocab.keys()
        self.vocab = self.d2v_model.wv.vocab.keys()
        # self.max_len = max_len

    def tokenizer(self, text):
        # tokens = tokenizer(text)
        filtered_tokens = []
        tokens = gensim.utils.simple_preprocess(text)
        doc_vec = self.d2v_model.infer_vector(tokens)
        vectors = []
        vectors.append(doc_vec)
        for word in tokens:
            if word in self.vocab:
                # filtered_tokens.append(word)
                vec = self.d2v_model[word]
                # vec = self.w2v_model[word]
                vectors.append(vec)
        # print(filtered_tokens)
        vectors = np.array(vectors)

        # if len(vectors) > self.max_len:
        #     vectors = vectors[0:self.max_len]
        #
        # if len(vectors) < self.max_len:
        #     padding = np.zeros((self.max_len - vectors.shape[0], 300))
        #     vectors = np.concatenate((vectors, padding), axis=0)
        #
        # doc_vec = np.expand_dims(doc_vec, axis=0)
        # vectors = np.concatenate((doc_vec, vectors), axis=0)
        vectors = torch.FloatTensor(vectors)

        return vectors, filtered_tokens


class Words2vecs():
    def __init__(self, model_path="./GoogleNews-vectors-negative300.bin.gz"):
        super(Words2vecs, self).__init__()
        self.model_path = model_path
        self.model = gensim.models.KeyedVectors.load_word2vec_format(self.model_path, binary=True)
        self.w2v_vocab = self.model.vocab.keys()

    def tokenizer(self, text):
        # tokens = tokenizer(text)
        tokens = gensim.utils.simple_preprocess(text)
        print(tokens)
        vectors = []
        for word in tokens:
            if word in self.w2v_vocab:
                vec = self.model[word]
                vectors.append(vec)

        vectors = np.array(vectors)

        if len(vectors) == 0:
            vectors = np.zeros((1, 300))
            # print("find zero-length text.")

        if len(vectors) > 196:
            vectors = vectors[0:196]

        if len(vectors) < 196:
            padding = np.zeros((196 - vectors.shape[0], 300))
            vectors = np.concatenate((vectors, padding), axis=0)

        vectors = torch.FloatTensor(vectors)

        return vectors

# class Words2vecs():
#
#     def __init__(self, model_path="./GoogleNews-vectors-negative300.bin.gz"):
#         super(Words2vecs, self).__init__()
#         self.model_path = model_path
#         self.model = gensim.models.KeyedVectors.load_word2vec_format(self.model_path, binary=True)
#         self.w2v_vocab = self.model.vocab.keys()
#         self.tok_vocab_k = list(tokenizer.vocab.keys())
#         self.tok_vocab_v = list(tokenizer.vocab.values())
#
#     def tokenizer(self, text):
#         if len(text.split(" ")) > 256:
#             words = text.split(" ")[0:256]
#             text = " ".join(words)
#
#         input_ids = tokenizer(text, return_tensors="pt", truncation=True).input_ids
#         vectors = []
#         for idx in input_ids[0]:
#             word = self.tok_vocab_k[self.tok_vocab_v.index(int(idx))]
#             if word in self.w2v_vocab:
#                 vec = self.model[word]
#                 vectors.append(vec)
#
#         vectors = np.array(vectors)
#
#         if len(vectors) == 0:
#             vectors = np.zeros((1, 300))
#
#         if len(vectors) > 196:
#             vectors = vectors[0:196]
#
#         if len(vectors) < 196:
#             padding = np.zeros((196 - vectors.shape[0], 300))
#             vectors = np.concatenate((vectors, padding), axis=0)
#
#         vectors = torch.FloatTensor(vectors)
#
#         return vectors
