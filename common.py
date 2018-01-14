import os
import re
from enum import Enum
import numpy as np


FLOAT_EPSILON = np.finfo(np.float).eps
NORMS = [1, 2, 'fro', 'inf']

DATA_FOLDER = os.path.join(os.path.dirname(__file__), 'AUTHORS')
TEMP_FOLDER = r'C:\temp'

CHARS_AS_WORDS = ',.();!?'


class TermFreqSchemes(Enum):
    RawCount = 1
    TermFrequency = 2


class AuthorWork:
    def __init__(self, data_folder, author_name, work_name):
        file_path = os.path.join(data_folder, author_name, work_name + '.txt')
        assert os.path.exists(file_path), file_path + ' incorrect'
        self.name = work_name
        self.author_name = author_name.upper()
        self.file_path = file_path
        self.raw_text = open(file_path, 'r').read()
        self.word_list = []

    def set_words(self, chars_as_words):
        pattern = re.compile('[^a-zA-Z0-9-]+')
        words = []
        for line in self.raw_text.splitlines():
            for word in line.split(' '):
                # remove blank spaces
                word = word.strip()
                if len(word) == 0:
                    continue
                word = word.lower()
                # treat selected characters as words
                for word_char in chars_as_words:
                    if word_char in word:
                        word.strip(word_char)
                        words.append(word_char)
                # remove non-alphanumeric characters
                word = pattern.sub('', word)
                if len(word) == 0:
                    continue
                words.append(word)
        self.word_list = words


class AuthorFolder:
    def __init__(self, data_folder, author_name):
        folder_name = author_name.upper()
        folder_path = os.path.join(data_folder, folder_name)
        assert os.path.exists(folder_path), folder_path + ' incorrect'
        author_files = os.listdir(folder_path)[:-1]
        self.name = folder_name
        self.folder = folder_path
        self.works_name_list = [work.split('.')[0] for work in author_files]
        self.works_path_list = [os.path.join(folder_path, file_name) for file_name in author_files]
        self.works_dict = dict(
            [(self.works_name_list[i], AuthorWork(data_folder, author_name, self.works_name_list[i]))
             for i in range(len(self.works_name_list))]
        )


AUTHORS_FOLDER_LIST = os.listdir(DATA_FOLDER)
AUTHORS_DICT = dict(
    [(author, AuthorFolder(DATA_FOLDER, author)) for author in AUTHORS_FOLDER_LIST]
)
ALL_WORKS = []
for author in AUTHORS_DICT.values():
    ALL_WORKS.extend(author.works_dict.values())


def compute_total_vocabulary(all_works, chars_as_words):
    all_words = []
    for work in all_works:
        work.set_words(chars_as_words)
        all_words.extend(work.word_list)
    return list(set(all_words))


def compute_term_frequencies(vocabulary, document_list, weight_scheme):
    # ftd_mat: [nbWordsInVocab][nbDocuments]
    ftd_mat = np.zeros([len(vocabulary), len(document_list)])
    # word_to_idx_map: {'word': row_index, ...} (for constant access)
    word_to_idx_map = dict([(word, idx) for idx, word in enumerate(vocabulary)])
    for doc_idx, document in enumerate(document_list):
        for word in document:
            term_idx = word_to_idx_map.get(word, None)
            if term_idx is not None:
                ftd_mat[term_idx, doc_idx] += 1

    for doc_idx in range(ftd_mat.shape[1]):
        assert ftd_mat[:, doc_idx].sum() == len(document_list[doc_idx]),\
            'len(ftd_mat) != len(document) at index %d' % doc_idx

    if weight_scheme == TermFreqSchemes.TermFrequency:
        for doc_idx, document in enumerate(document_list):
            ftd_mat[:, doc_idx] /= len(document)

    return ftd_mat


def compute_relevance_vector(all_docs_freq_matrix, query_freq_vector, norm):
    query_norm = np.linalg.norm(query_freq_vector, norm)
    doc_norm_vec = np.array([np.linalg.norm(col, norm) for col in all_docs_freq_matrix.T])
    relevance_vec = all_docs_freq_matrix.T.dot(query_freq_vector)
    relevance_vec = np.multiply(relevance_vec.T, doc_norm_vec)
    relevance_vec /= query_norm
    return relevance_vec[0]


def compute_inverse_doc_frequencies(ftd_mat):
    # idf_vec: [nbWordsInVocab]
    nb_words, nb_docs = ftd_mat.shape[:]
    idf_vec = np.full([nb_words], FLOAT_EPSILON)
    for term_idx in range(nb_words):
        idf_vec[term_idx] += np.count_nonzero(ftd_mat[term_idx])

    # for idf in idf_vec:
    #     assert 0 < idf <= nb_docs, 'document frequency not within [0,nbDocs] bounds: %d' % idf

    return np.apply_along_axis(lambda x: np.log(nb_docs / x), 0, idf_vec)


def main():

    total_vocab = compute_total_vocabulary(ALL_WORKS, CHARS_AS_WORDS)
    print('Total Vocabulary length: %d' % len(total_vocab))
    with open(os.path.join(TEMP_FOLDER, 'total_vocab.txt'), 'w') as f:
        f.write('\n'.join(total_vocab))
    all_documents = [work.word_list for work in ALL_WORKS]

    ftd_mat = compute_term_frequencies(total_vocab, all_documents, TermFreqSchemes.TermFrequency)
    idf_vec = compute_inverse_doc_frequencies(ftd_mat)
    tfidf_mat = np.apply_along_axis(lambda col: np.multiply(col, idf_vec), 0, ftd_mat)

    test_query = AuthorWork(DATA_FOLDER, 'SHAKESPEARE', 'synopsislst')
    test_query.set_words(CHARS_AS_WORDS)
    test_query_ftd = compute_term_frequencies(total_vocab, [test_query.word_list], TermFreqSchemes.TermFrequency)
    # test_query_idf = compute_inverse_doc_frequencies(test_query_ftd)
    # test_query_tfidf = np.apply_along_axis(lambda col: np.multiply(col, test_query_idf), 0, test_query_ftd)

    test_query_relevance_vec = compute_relevance_vector(tfidf_mat, test_query_ftd, 2)
    work_score_vec = [(work.name, work.author_name, score)
                      for work, score in zip(ALL_WORKS, test_query_relevance_vec)]
    test_query_relevance_ranking = sorted(work_score_vec, key=lambda x: x[-1], reverse=True)
    for line in test_query_relevance_ranking:
        print('%s (%s): %f' % line[:])

    return

if __name__ == '__main__':
    main()
