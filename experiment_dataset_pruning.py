import numpy as np
from sklearn.cluster import MiniBatchKMeans
from typing import Optional
from .classification import experiment, logreg
from .utils import RANDOM_STATE, set_seed, standardize_data


def run_experiment_on_sample(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        subset_idx: Optional[Iterable[int]] = None,
        sample_size: Optional[int] = None,
        replace: bool = False,
        n_repeat: int = 1,
        random_state: int = RANDOM_STATE,
        logreg_config: dict = None,
        verbose: bool = False,
    ):
    """ Runs experiment with optimal subset of objects (passed as subset_idx, or samples itself).
        :params X_train, y_train, X_test, y_test: data to train and evaluate on, X being of shape <n_layers, n_samples, n_dim>
        :param subset_idx: list of integers forming indicies to be taken from X_train and y_train. If None, sample_sizes are sampled randomly
        :param sample_size: integer indicating how many samples to take from X_train and y_train. Ignored if indicies are passed explicitly in subset_idx
        :param replace: bootstrap if True, random sampling if False. Ignored if indicies are passed explicitly in subset_idx
        :param n_repeat: how many times to run the experiment. Use > 1 to reduce noise
        :param random_state: random state
        :param logreg_config: config for logistic regression
        :param verbose: show progress if True, silent of False
        :return: dictionary of scores per each layer
    """
    scores = {}

    resample_data = subset_idx is None
    config_set = logreg_config is not None

    X_train_sampled = X_train[:, subset_idx] if not resample_data else None
    y_train_sampled = y_train[subset_idx] if not resample_data else None

    all_scores = []  # repeated exps
    for i in range(n_repeat):

        set_seed(random_state + i)

        if resample_data:
            # we need to generate sample automatically
            subset_idx = np.random.choice(X_train.shape[1], sample_size, replace=replace)        
            X_train_sampled = X_train[:, subset_idx]
            y_train_sampled = y_train[subset_idx]

        config = {
            'solver': 'sag',
            'tol': 0.01,
            'max_iter': 200,
            'random_state': random_state + i,
            # 'verbose': True,
            # 'n_iter_no_change': 5,
            # 'fit_intercept': False,
        } if not config_set else logreg_config

        # run experiment
        all_scores.append(experiment(logreg, config, X_train_sampled, y_train_sampled, X_test, y_test, all_layers=True, verbose=verbose))
        # del all_scores[-1]['lrs']

    # free memory explicitly
    del X_train_sampled
    del y_train_sampled
    gc.collect()
    
    # collect all scores into lists
    scores = {}
    for key in list(all_scores[0].keys()):
        scores[key] = [sc[key] for sc in all_scores]
    return scores


def run_experiment_subsample_vs_kmeans(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        target_sz: int,
        n_repeat: int = 10,
        random_state: int = RANDOM_STATE,
        verbose: bool = False
    ) -> Tuple[dict, dict]:
    """ Runs experiment to compare random sampling strategy vs kmeans pruning.
        :params X_train, y_train, X_test, y_test: data to train and evaluate on, X being of shape <n_layers, n_samples, n_dim>
        :param target_sz: number of objects to pick
        :param n_repeat: n_repeat for run_experiment_on_sample
        :param random_state: random_state
        :param verbose: show progress if True, silent of False
        :return: pair of dict: scores from random sampling and scores from kmeans sampling
    """
    # random sample
    if verbose:
        print('preparing random sample...')
    set_seed(random_state)
    random_idx = np.random.choice(X_train.shape[1], target_sz, replace=False)
    random_score = run_experiment_on_sample(X_train, y_train, X_test, y_test, sample_size=target_size, n_repeat=n_repeat, verbose=verbose)

    if verbose:
        print('training kmeans...')
    # kmeans sample
    kmeans = MiniBatchKMeans(n_clusters=target_sz, verbose=verbose)
    start = time()
    clusters = kmeans.fit_predict(X_train[-1])
    # with open('clusters.txt', 'w') as f:
    #     json.dump(clusters.tolist(), f)
    if verbose:
        print('training took time: {:.2f} seconds'.format(time() - start))
        print('getting kmeans indicies...')
    kmeans_idxs = []  # indexes chosen by K-Means
    weights = []  # assigned to size of cluster
    for cluster in np.unique(clusters):
        cluster_objects = np.where(clusters == cluster)[0]
        kmeans_idxs.append(np.random.choice(cluster_objects))  # chosen object
        weights.append(len(cluster_objects))
    kmeans_score = run_experiment_on_sample(X_train, y_train, X_test, y_test, subset_idx=kmeans_idxs)
    return random_score, kmeans_score


""" Usage example: """

if __name__ == '__main__':
    folder = 'TenseBert128_mean'
    X_train = np.load('/content/gdrive/My Drive/Thesis/{}/X_train.npy'.format(folder))
    y_train = np.load('/content/gdrive/My Drive/Thesis/{}/y_train.npy'.format(folder))
    X_test = np.load('/content/gdrive/My Drive/Thesis/{}/X_test.npy'.format(folder))
    y_test = np.load('/content/gdrive/My Drive/Thesis/{}/y_test.npy'.format(folder))
    X_train, X_test = standardize_data(X_train, X_test)
    
    # random sampling to different sample_sizes and storing scores
    sample_sizes = np.unique(np.clip(np.round(np.logspace(1, 5, 13), -2).astype(np.int32), a_min=100, a_max=y_train.shape[0]))  # [100, 200, 500, 1000, 2200, 4600, 10000, 21500, 46400, 100000]
    subset_idxs = []
    scores = {}
    for sample_size in tqdm(sample_sizes, desc='Sample size'):
        print('=== sample size: %d' % sample_size)
        # building subset_idx for each layer
        scores[str(sample_size)] = run_experiment_on_sample(X_train, y_train, X_test, y_test, sample_size=sample_size, n_repeat=10)
    json.dump(scores, 'random sampling probing results.txt')

    # comparing random sampling to kmeans pruning
    # best scores:
    golden_score = run_experiment_on_sample(X_train, y_train, X_test, y_test, subset_idx=np.arange(X_train.shape[1]))
    # down-sampling once for faster experiment:
    set_seed(34090)
    subset_idx = np.random.choice(X_train.shape[1], 10000, replace=False)
    sample_score = run_experiment_on_sample(X_train, y_train, X_test, y_test, subset_idx)
    # running comparison
    random_scores = []
    kmeans_scores = []
    for target_sz in [200, 400, 600, 800]:
        random_score, kmeans_score = run_experiment_subsample_vs_kmeans(X_train[:, subset_idx], y_train[subset_idx], X_test, y_test, target_sz=target_sz, n_repeat=10verbose=False)
        random_scores.append(random_score)
        kmeans_scores.append(kmeans_score)
    with open('random_vs_kmeans.txt', 'w') as f:
        json.dump(random_scores, f)
        json.dump(kmeans_scores, f)
