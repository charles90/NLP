import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
import bisect
import time
import common


class RunOptions(object):
    def __init__(self, flags=None):
        if flags is None:
            return
        self.words_file_path = flags.words_file_path
        self.reasoning_file_path = flags.reasoning_file_path
        self.output_file_path = flags.output_file_path
        self.debug = flags.debug
        self.confidence = flags.confidence
        self.nb_noises = flags.nb_noises
        self.nb_features = flags.nb_features
        self.window_size = flags.window_size
        self.batch_size = flags.batch_size
        self.learning_rate = flags.learning_rate
        self.nb_steps = flags.nb_steps
        self.occurrence_threshold = flags.occurrence_threshold

    def __str__(self):
        this_str = "%s: %d\n" % ('nb_noises', self.nb_noises)
        this_str += "%s: %d\n" % ('nb_features', self.nb_features)
        this_str += "%s: %d\n" % ('window_size', self.window_size)
        this_str += "%s: %d\n" % ('batch_size', self.batch_size)
        this_str += "%s: %.4f\n" % ('learning_rate', self.learning_rate)
        this_str += "%s: %d\n" % ('nb_steps', self.nb_steps)
        this_str += "%s: %d\n" % ('occurrence_threshold', self.occurrence_threshold)
        return this_str

    def __iter__(self):
        yield 'nb_noises', self.nb_noises
        yield 'nb_features', self.nb_features
        yield 'window_size', self.window_size
        yield 'batch_size', self.batch_size
        yield 'learning_rate', self.learning_rate
        yield 'nb_steps', self.nb_steps
        yield 'occurrence_threshold', self.occurrence_threshold


def generate_batch(words, vocab_dict, batch_size, window_size):
    unknown_idx = vocab_dict['<UNK>']
    train_idx_list, label_idx_list = [], []
    for idx in range(batch_size):
        word_idx = np.random.randint(window_size, len(words) - window_size)
        word = words[word_idx]
        label_idx = vocab_dict.get(word, unknown_idx)
        label_idx_list.append([label_idx])
        contexts = list(range(word_idx - window_size, word_idx)) + list(range(word_idx + 1, word_idx + window_size + 1))
        train_idx_list.append([vocab_dict.get(words[corpus_idx], unknown_idx) for corpus_idx in contexts])
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


def run_model(options):

    with open(options.words_file_path) as f:
        all_words = tf.compat.as_str_any(f.read()).split()

    total_vocab = list(set(all_words))
    occur_vec = common.compute_term_frequencies(total_vocab, [all_words], common.TermFreqSchemes.RawCount)
    reduced_vocab = ['<UNK>']
    for idx, word in enumerate(total_vocab):
        if occur_vec[idx] >= options.occurrence_threshold:
            reduced_vocab.append(word)
    vocab_size = len(reduced_vocab)
    word_to_idx_dict = dict([(word, idx) for idx, word in enumerate(reduced_vocab)])

    words_1, words_2, words_3, words_4 = gather_analogical_vectors(options.reasoning_file_path, word_to_idx_dict)

    graph = tf.Graph()

    with graph.as_default():

        train_indices = tf.placeholder(tf.int32,    [options.batch_size, 2 * options.window_size])
        label_indices = tf.placeholder(tf.int32,    [options.batch_size, 1])
        sampled_indices = tf.placeholder(tf.int32,  [options.batch_size, 2 * options.window_size * options.nb_noises])

        with tf.device('/gpu:0'):

            embeddings = tf.Variable(tf.random_uniform([vocab_size, options.nb_features], -1.0, 1.0))
            train_embeddings = tf.nn.embedding_lookup(embeddings, train_indices)
            label_embeddings = tf.nn.embedding_lookup(embeddings, label_indices)
            sampled_embeddings = tf.nn.embedding_lookup(embeddings, sampled_indices)

            nec_activation = tf.nn.sigmoid(-1 * tf.matmul(sampled_embeddings, label_embeddings, transpose_b=True))
            log_nec_activation = tf.log(nec_activation)
            nec_loss = tf.reduce_mean(log_nec_activation)

            embed_activation = tf.nn.sigmoid(tf.matmul(train_embeddings, label_embeddings, transpose_b=True))
            log_embed_activation = tf.log(embed_activation)
            loss_function = -1 * (tf.reduce_mean(log_embed_activation) + nec_loss)

            trainer = tf.train.GradientDescentOptimizer(options.learning_rate).minimize(loss_function)

            init = tf.global_variables_initializer()

    t0 = time.time()
    losses = []

    with tf.Session(graph=graph) as session:

        if options.debug is True:
            session = tf_debug.LocalCLIDebugWrapperSession(session)
            session.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        init.run()
        for step in range(options.nb_steps):
            batch_indices, batch_labels = generate_batch(
                all_words,
                word_to_idx_dict,
                options.batch_size,
                options.window_size)
            batch_samples = generate_word_sample(
                word_to_idx_dict,
                options.nb_noises,
                options.batch_size,
                options.window_size)

            err, loss = session.run([trainer, loss_function],
                                    {
                                        train_indices: batch_indices,
                                        label_indices: batch_labels,
                                        sampled_indices: batch_samples
                                    })
            losses.append(loss)
            if step % 500 == 0:
                print(step)
                print(loss)
            if err is not None:
                print(err)

        train_time = time.time() - t0
        smallest_loss = min(losses)
        last_loss = losses[-1]
        loss_stddev = np.std(losses)
        loss_stddev_100 = np.std(losses[-100:])
        print('Total training time: %d seconds' % train_time)
        print('Smallest loss: %f' % smallest_loss)
        print('Last loss: %f' % last_loss)
        print('Standard Deviation: %f' % loss_stddev)
        print('Standard Deviation in the last 100 steps: %f' % loss_stddev_100)

        logical_diffs = evaluate_analogical_reasoning(words_1, words_2, words_3, words_4, embeddings).eval()
        avg_diff = tf.reduce_mean(logical_diffs).eval()
        min_dist = min(logical_diffs)
        max_dist = max(logical_diffs)
        print()
        print('Average L2 analogical reasoning distance: %f' % avg_diff)
        print('Minimum L2 analogical reasoning distance: %f' % min_dist)
        print('Maximum L2 analogical reasoning distance: %f' % max_dist)

        coverage = 0
        for dist in logical_diffs:
            if (1 - dist) >= options.confidence:
                coverage += 1
        coverage /= len(logical_diffs)
        print()
        print('Coverage with confidence %.2f%%: %f' % (options.confidence*100, coverage))

        results_dict = {
            "train_time": train_time,
            "smallest_loss": smallest_loss,
            "last_loss": last_loss,
            "loss_stddev": loss_stddev,
            "loss_stddev_100": loss_stddev_100,
            "avg_diff": avg_diff,
            "min_dist": min_dist,
            "max_dist": max_dist,
            "coverage": coverage
        }
        params_dict = {
            "nb_noises": options.nb_noises,
            "nb_features": options.nb_features,
            "window_size": options.window_size,
            "batch_size": options.batch_size,
            "learning_rate": options.learning_rate,
            "nb_steps": options.nb_steps
        }
        if options.output_file_path is not None:
            measures = list(params_dict.keys())
            measures.append("coverage")
            values = [str(val) for val in params_dict.values()]
            values.append(str(coverage))
            with open(options.output_file_path, "a") as f:
                f.write(','.join(measures) + '\n')
                f.write(','.join(values) + '\n')

    return results_dict


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
    flags.DEFINE_string('output_file_path', None, 'Output path for various results of a run')
    flags.DEFINE_float('confidence', 0.95, 'Level of confidence required for the word similarity to be approved')
    flags.DEFINE_integer('occurrence_threshold', 5, 'Minimum number of occurrences to be included in the vocabulary')
    options = RunOptions(flags.FLAGS)
    run_model(options)


if __name__ == '__main__':
    main()
