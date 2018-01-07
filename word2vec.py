import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
import bisect
import time
import common


def generate_batch(words, vocab_dict, batch_size, window_size):
    train_idx_list, label_idx_list = [], []
    for idx in range(batch_size):
        word_idx = np.random.randint(window_size, len(words) - window_size)
        word = words[word_idx]
        label_idx_list.append([vocab_dict[word]])
        contexts = list(range(word_idx - window_size, word_idx)) + list(range(word_idx + 1, word_idx + window_size + 1))
        train_idx_list.append([vocab_dict[words[corpus_idx]] for corpus_idx in contexts])
    return train_idx_list, label_idx_list


def generate_word_sample(vocab_dict, nb_samples, batch_size, window_size, ordered_probas=None):
    vocab_length = len(vocab_dict.keys())
    if ordered_probas is not None:
        assert vocab_length == len(ordered_probas), \
            'vocab_length != ordered_probas length: %d vs %d' % (vocab_length, len(ordered_probas))
        uniform_draws = np.random.rand(batch_size, 2 * window_size * nb_samples)
        sample_idx_list = np.empty_like(uniform_draws, dtype=np.int32)
        for iRow in range(sample_idx_list.shape[0]):
            for iCol in range(sample_idx_list.shape[1]):
                sample_idx_list[iRow, iCol] = bisect.bisect_left(ordered_probas, uniform_draws[iRow, iCol], hi=len(ordered_probas)-1)
    else:
        sample_idx_list = np.random.randint(0, vocab_length, (batch_size, 2 * window_size * nb_samples))
    return sample_idx_list


def gather_analogical_vectors(q_and_a_file_path, vocab_dict):
    # line format:
    # Berlin Germany Paris France
    words_1, words_2, words_3, words_4 = [], [], [], []
    with open(q_and_a_file_path) as f:
        for line in f.readlines():
            if line.startswith(':'):
                continue
            split_line = line.lower().split()
            try:
                key_1 = vocab_dict[split_line[0]]
                key_2 = vocab_dict[split_line[1]]
                key_3 = vocab_dict[split_line[2]]
                key_4 = vocab_dict[split_line[3]]
            except KeyError:
                continue
            words_1.append(key_1)
            words_2.append(key_2)
            words_3.append(key_3)
            words_4.append(key_4)
    return words_1, words_2, words_3, words_4


def evaluate_analogical_reasoning(words_1, words_2, words_3, words_4, embeddings):
    # Berlin Germany Paris France
    # Vec('Berlin') - Vec('Germany') == Vec('Paris') - Vec('France')
    # comparison is done using cosine distance
    embed_1 = tf.nn.embedding_lookup(embeddings, words_1)
    embed_2 = tf.nn.embedding_lookup(embeddings, words_2)
    embed_3 = tf.nn.embedding_lookup(embeddings, words_3)
    embed_4 = tf.nn.embedding_lookup(embeddings, words_4)
    lhs_vec = tf.subtract(embed_1, embed_2)
    rhs_vec = tf.subtract(embed_3, embed_4)
    embed_diff = tf.reduce_sum(tf.multiply(lhs_vec, rhs_vec), axis=1)
    embed_diff /= tf.norm(lhs_vec, axis=1) * tf.norm(rhs_vec, axis=1)
    return tf.abs(embed_diff)


def main():

    flags = tf.app.flags
    flags.DEFINE_string('words_file_path', None, 'File path to a list of space-separated words')
    flags.DEFINE_string('reasoning_file_path', None, 'File path to a 4-word analogical reasoning task per row')
    flags.DEFINE_integer('nb_noises', 5, 'Number of noises for the Negative Sampling')
    flags.DEFINE_integer('nb_features', 56, 'Length of word embedding vector')
    flags.DEFINE_integer('window_size', 3, 'Number of context words on either side of target word')
    flags.DEFINE_integer('batch_size', 512, 'Size of input for each step of Stochastic Gradient Descent')
    flags.DEFINE_float('learning_rate', 1.0, 'Constant learning rate of Stochastic Gradient Descent')
    flags.DEFINE_integer('nb_steps', 1000, 'Number of Stochastic Gradient Descent steps')
    flags.DEFINE_boolean('debug', False, 'Activate TensorFlow debug mode')
    options = flags.FLAGS

    with open(options.words_file_path) as f:
        all_words = tf.compat.as_str_any(f.read()).split()

    total_vocab = list(set(all_words))
    vocab_size = len(total_vocab)
    word_to_idx_dict = dict([(word, idx) for idx, word in enumerate(total_vocab)])
    freq_vec = common.compute_term_frequencies(total_vocab, [all_words], common.TermFreqSchemes.TermFrequency)

    words_1, words_2, words_3, words_4 = gather_analogical_vectors(options.reasoning_file_path, word_to_idx_dict)

    graph = tf.Graph()

    with graph.as_default():

        train_indices = tf.placeholder(tf.int32,    [BATCH_SIZE, 2 * WINDOW_SIZE])
        label_indices = tf.placeholder(tf.int32,    [BATCH_SIZE, 1])
        sampled_indices = tf.placeholder(tf.int32,  [BATCH_SIZE, 2 * WINDOW_SIZE * NB_NOISES])

        with tf.device('/gpu:0'):

            embeddings = tf.Variable(tf.random_uniform([vocab_size, NB_FEATURES], -1.0, 1.0))
            train_embeddings = tf.nn.embedding_lookup(embeddings, train_indices)
            label_embeddings = tf.nn.embedding_lookup(embeddings, label_indices)
            sampled_embeddings = tf.nn.embedding_lookup(embeddings, sampled_indices)

            nec_activation = tf.nn.sigmoid(-1 * tf.matmul(sampled_embeddings, label_embeddings, transpose_b=True))
            log_nec_activation = tf.log(nec_activation)
            nec_loss = tf.reduce_mean(log_nec_activation)

            embed_activation = tf.nn.sigmoid(tf.matmul(train_embeddings, label_embeddings, transpose_b=True))
            log_embed_activation = tf.log(embed_activation)
            loss_function = -1 * (tf.reduce_mean(log_embed_activation) + nec_loss)

            trainer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss_function)

            init = tf.global_variables_initializer()

    t0 = time.time()
    losses = []

    with tf.Session(graph=graph) as session:

        if options.debug is True:
            session = tf_debug.LocalCLIDebugWrapperSession(session)
            session.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        init.run()
        for step in range(NB_STEPS):
            batch_indices, batch_labels = generate_batch(all_words, word_to_idx_dict, BATCH_SIZE, WINDOW_SIZE)
            batch_samples = generate_word_sample(word_to_idx_dict, NB_NOISES, BATCH_SIZE, WINDOW_SIZE)

            err, loss = session.run([trainer, loss_function],
                                    {
                                        train_indices: batch_indices,
                                        label_indices: batch_labels,
                                        sampled_indices: batch_samples
                                    })
            losses.append(loss)
            print(step)
            print(loss)
            if err is not None:
                print(err)

        print('Total training time: %d seconds' % (time.time() - t0))
        print('Smallest loss: %f' % min(losses))
        print('Last loss: %f' % losses[-1])
        print('Standard Deviation: %f' % np.std(losses))
        print('Standard Deviation in the last 100 steps: %f' % np.std(losses[-100:]))

        logical_diffs = evaluate_analogical_reasoning(words_1, words_2, words_3, words_4, embeddings)
        avg_diff = tf.reduce_mean(logical_diffs)
        print()
        print('Average L2 analogical reasoning distance: %f' % avg_diff.eval())
        print('Minimum L2 analogical reasoning distance: %f' % min(logical_diffs.eval()))
        print('Maximum L2 analogical reasoning distance: %f' % max(logical_diffs.eval()))

if __name__ == '__main__':
    main()
