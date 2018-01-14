import numpy as np
import pandas as pd

import word2vec

NOISES_RANGE = [1, 25]
FEATURES_RANGE = [10, 128]
WINDOW_SIZE_RANGE = [1, 10]
BATCH_SIZE_RANGE = [100, 1000]
LEARN_RATE_RANGE = [1, 4]

LOG_DISTRIBUTIONS = ["learning_rate"]
NB_RUNS = 20
METRIC_LABEL = 'coverage'

options = word2vec.RunOptions({})
options.words_file_path = r"C:\workspace\NLP\text8\text8"
options.reasoning_file_path = r"C:\workspace\NLP\source-archive\word2vec\trunk\questions-words.txt"
options.output_file_path = r"C:\temp\hyper_param_random_run.csv"
options.debug = False
options.confidence = 0.95

result_records = []

for run_idx, nb_noises, nb_features, window_size, batch_size, learning_rate in zip(
    range(0, NB_RUNS),
    np.random.randint(NOISES_RANGE[0], NOISES_RANGE[1]+1, NB_RUNS),
    np.random.randint(FEATURES_RANGE[0], FEATURES_RANGE[1]+1, NB_RUNS),
    np.random.randint(WINDOW_SIZE_RANGE[0], WINDOW_SIZE_RANGE[1]+1, NB_RUNS),
    np.random.randint(BATCH_SIZE_RANGE[0], BATCH_SIZE_RANGE[1]+1, NB_RUNS),
    np.power(10, -1 * (np.random.rand(NB_RUNS) * (LEARN_RATE_RANGE[1] - LEARN_RATE_RANGE[0]) + LEARN_RATE_RANGE[0]))
):
    options.nb_noises = nb_noises
    options.nb_features = nb_features
    options.window_size = window_size
    options.batch_size = batch_size
    options.learning_rate = learning_rate
    options.nb_steps = 1000
    print()
    print("Run number %d" % run_idx)
    print(str(options))

    results_dict = word2vec.run_model(options)
    param_run_dict = dict(options)
    param_run_dict[METRIC_LABEL] = results_dict[METRIC_LABEL]
    result_records.append(param_run_dict)

print()
res_df = pd.DataFrame.from_records(result_records)
for col in res_df.columns:
    if col == METRIC_LABEL or col == 'nb_steps':
        continue
    corr = res_df[col].corr(res_df[METRIC_LABEL])
    print("Parameter %s has %d%% correlation with %s" % (col, corr*100, METRIC_LABEL))
