# Функции для определения feature importance

import numpy as np
from scipy.stats import ttest_ind
from skfeature.function.similarity_based import reliefF, fisher_score, trace_ratio, SPEC, lap_score  # please see https://github.com/jundongl/scikit-feature
from typing import Callable, Iterable, Optional
from .utils import RANDOM_STATE, set_seed


def statistical_selection(X: np.ndarray, y: np.ndarray, ndim: Optional[int] = None, stat: str = 't-score') -> np.array:
    """ Returns indicies of optimal features to select from X using statistical approach. 
        :param X: objects dataset, np.ndarray of size <n_objects, n_features>
        :param y: targets dataset, np.array of size n_objects
        :param ndim: the number of features to select. If None, all features are returned in order of their importance
        :param stat: which statistical method to use
        :return: np.array of most important feature indicies in order of their importance
    """
    if ndim is None:
        ndim = X.shape[-1]

    scores = []
    
    if stat == 't-score':  # t-test selection
        if len(np.unique(y)) > 2:
            raise ValueError('Can\'t apply binary t-test to multiclass labels. Try xi2-score instead.')
        X1 = X[y == 0]
        X2 = X[y == 1]
        scores, _ = ttest_ind(X1, X2, axis=0, equal_var=False)
    elif stat == 'var':  # low variance selection
        scores = np.var(X, axis=0)
    else:
        raise NotImplementedError('Not implemented statistic {} in statistical_selection!'.format(stat))
    
    importance_order = np.argsort(scores)[::-1]
    return importance_order[:ndim]


def similarity_selection(X: np.ndarray, y: np.ndarray, similarity_callable: Callable, ndim: Optional[int] = None, sample_size: int = 1000, random_state: int = RANDOM_STATE) -> np.array:
    """ Returns indicies of optimal features to select from sample of X using similarity-based approach. """
    if ndim is None:
        ndim = X.shape[-1]
    set_seed(random_state)
    sample_idx = np.random.choice(X.shape[0], sample_size)
    scores = similarity_callable(X[sample_idx], y[sample_idx])
    importance_order = np.argsort(scores)[::-1]
    return importance_order[:ndim]


def relieff_selection(*args, **kwargs) -> np.array:
    return similarity_selection(*args, **kwargs, similarity_callable=reliefF.reliefF)


def fisher_selection(X: np.ndarray, y: np.ndarray, ndim: Optional[int] = None, sample_size: int = 1000):
    return similarity_selection(*args, **kwargs, similarity_callable=fisher_score.fisher_score)


def make_list_of_feature_orders(X: np.ndarray, y: np.ndarray, order_func: Callable, *args, **kwargs) -> List[np.array]:
    """ Returns list of lists of feature importance orders for each layer. Helper function for run_experiment_with_selected_features arguments.
        :param X: objects dataset, np.ndarray of size <n_layers, n_objects, n_features>
        :param y: targets dataset, np.array of size n_objects
        :param order_func: feature selection function
        :params *args, **kwargs: passed to order_func
        :return: list of feature orders for each layer
    """
    feature_orders = []
    for layer in range(X.shape[0]):
        feature_orders.append(order_func(X[layer], y, *args, **kwargs))
    return feature_orders


def run_experiment_with_selected_features(X_train, y_train, X_test, y_test, feature_orders: Iterable[Iterable], step: int = 1, verbose: bool = False, random_state: int = RANDOM_STATE) -> dict:
    """ Runs experiment with optimal subset of features (passed as feature_orders for each layer).
        :params X_train, y_train, X_test, y_test: data to train and evaluate on, X being of shape <n_layers, n_samples, n_dim>
        :param feature_orders: list of feature orders for each layer. Compatible with returns of make_list_of_feature_orders
        :param step: step of experiment grid (see ndims in code)
        :param verbose: show progress if True, silent of False
        :param random_state: random state
        :return: scores per each ndim
    """
    scores = {}

    ndims = list(range(1, X_train.shape[2] + 1, step))
    if X_train.shape[-1] not in ndims:
        ndims.append(X_train.shape[-1])
        
    if verbose:
        ndims = tqdm(ndims, desc='ndim')
    
    for ndim in ndims:
        config = {
            'solver': 'sag',
            'tol': 0.01,
            'max_iter': 100,
            'random_state': random_state,
        }

        # remove the least important features
        X_trains_reduced = []  # holder for reduced train data
        X_tests_reduced = []  # holder for reduced test data
        for layer in range(X_train.shape[0]):
            X_train_i = X_train[layer][:, feature_orders[layer][:ndim]]
            X_test_i = X_test[layer][:, feature_orders[layer][:ndim]]
            X_trains_reduced.append(X_train_i)
            X_tests_reduced.append(X_test_i)
        X_train_reduced = np.stack(X_trains_reduced)
        X_test_reduced = np.stack(X_tests_reduced)

        # free memory explicitly
        del X_trains_reduced
        del X_tests_reduced
        gc.collect()

        # run experiment
        scores[ndim] = experiment(logreg, config, X_train_reduced, y_train, X_test_reduced, y_test, all_layers=True, verbose=verbose)
        del scores[ndim]['lrs']

        # free memory explicitly
        del X_train_reduced
        del X_test_reduced
        gc.collect()

    return scores


if __name__ == '__main__':
    # ... load probing data and standardize X_train, X_test

    # statistical: t-score selection
    t_scores = run_experiment_with_selected_features(X_train, y_train, X_test, y_test, step=4,
                                                   feature_orders=make_list_of_feature_orders(X_train, y_train, statistical_selection, stat='t-score'))
    # statistical: var selection
    var_scores = run_experiment_with_selected_features(X_train, y_train, X_test, y_test, step=4,
                                                   feature_orders=make_list_of_feature_orders(X_train, y_train, statistical_selection, stat='var'))

    # similarity-based: fisher selection
    fisher_scores = run_experiment_with_selected_features(X_train, y_train, X_test, y_test, step=4,
                                                          feature_orders=make_list_of_feature_orders(X_train, y_train, fisher_selection))
    
    # similarity-based: relieff selection
    relieff_scores = run_experiment_with_selected_features(X_train, y_train, X_test, y_test, step=4,
                                                           feature_orders=make_list_of_feature_orders(X_train, y_train, relieff_selection))
    
    with open('feature_selection.txt', 'w') as f:
        json.dump({
            't-score': t_scores,
            'var': var_scores,
            'fisher': fisher_scores,
            'relieff': relieff_scores,
        }, f)
