import json
import matplotlib.pyplot as plt
import numpy as np
import sys
from matplotlib.figure import Figure
from numpy.typing import ArrayLike
from scipy.stats import pearsonr, spearmanr
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from time import time, sleep
from typing import Any, Callable, Dict, Optional, Type



def logreg(X_train: ArrayLike, y_train: ArrayLike, X_test: ArrayLike, y_test: ArrayLike, classifier_config: Dict[str, Any]) -> ClassifierMixin:
    """
        Обучает логрег из sklearn на выборке эмбеддов <X_train, y_train>
        возвращает модель для последующего снятия скоров
    """
    lr = LogisticRegression(**classifier_config)
    lr = lr.fit(X_train, y_train)
    return lr


def sgd(X_train: ArrayLike, y_train: ArrayLike, X_test: ArrayLike, y_test: ArrayLike, classifier_config: Dict[str, Any]) -> ClassifierMixin:
    """
        Аналогично, обучает sgd из sklearn на выборке эмбеддов <X_train, y_train>
        возвращает модель для последующего снятия скоров
    """
    sgd = SGDClassifier(**classifier_config)
    sgd = sgd.fit(X_train, y_train)
    return sgd


def sklearn_score(lr: ClassifierMixin, X_train: ArrayLike, y_train: ArrayLike, X_test: ArrayLike, y_test: ArrayLike, binary: bool = True) -> Tuple[float, float, float, float, float]:
    """
        Снимает скоры с sklearn-классификатора lr, возвращает
        :param lr: classifier
        :param binary: whether the target is binary
        :return: f1_train, roc_auc_train, acc_train, f1_test, roc_auc_test, acc_test, n_iter
    """
    average = 'binary' if binary else 'weighted'
    y_train_pred = lr.predict(X_train)
    y_train_pred_proba = lr.predict_proba(X_train)
    y_test_pred = lr.predict(X_test)
    y_test_pred_proba = lr.predict_proba(X_test)
    if binary:
        y_train_pred_proba = y_train_pred_proba[:, 1]
        y_test_pred_proba = y_test_pred_proba[:, 1]


    roc_auc_params = {
        'average': 'weighted' if not binary else None,
        'multi_class': 'ovr' if not binary else 'raise',
    }

    f1_train = f1_score(y_train, y_train_pred, average=average)
    roc_auc_train = roc_auc_score(y_train, y_train_pred_proba, **roc_auc_params)
    pr_auc_train = 0 if not binary else average_precision_score(y_train, y_train_pred_proba)
    acc_train = accuracy_score(y_train, y_train_pred)
    f1_test = f1_score(y_test, y_test_pred, average=average)
    roc_auc_test = roc_auc_score(y_test, y_test_pred_proba, **roc_auc_params)
    pr_auc_test = 0 if not binary else average_precision_score(y_test, y_test_pred_proba)
    acc_test = accuracy_score(y_test, y_test_pred)
    n_iter = lr.n_iter_ if type(lr.n_iter_) == int else lr.n_iter_.item()
    return f1_train, roc_auc_train, pr_auc_train, acc_train, f1_test, roc_auc_test, pr_auc_test, acc_test, n_iter


# Sklearn probing experiment

def experiment(
    classifier_func: Callable,
    classifier_config: Dict[str, Any] = {},
    # model_name=None,
    # model=None,
    # tokenizer=None,
    # model_class=BertModel,
    # tokenizer_class=BertTokenizer,
    X_train: Optional[ArrayLike] = None,
    y_train: Optional[ArrayLike] = None,
    X_test: Optional[ArrayLike] = None,
    y_test: Optional[ArrayLike] = None,
    # dataset=None,
    # dataset_X=None,
    # dataset_y=None,
    # sentence_embeddings_func=get_sentence_embs,
    all_layers: bool = True,
    verbose: bool = True,
):
    """ Runs one probing classification experiment with given parameters. Parameters for the classifier are set in the config dict """
    if verbose:
        print('Classifier: {}, {}'.format(classifier_func.__name__, classifier_config))
    layers_cnt = X_train.shape[0]
    is_binary = True if len(np.unique(y_train)) == 2 else False

    if verbose:
        print('probing...')
    layers = list(range(layers_cnt)) if all_layers else [layers_cnt - 1]
    scores = {
        'f1_train': [],
        'f1_test': [],
        'roc_auc_train': [],
        'roc_auc_test': [],
        'pr_auc_train': [],
        'pr_auc_test': [],
        'acc_train': [],
        'acc_test': [],
        'training_time': [],
        'n_iter': [],
        'lrs': [],
    }
    for layer in layers:
        start_time = time()
        lr = classifier_func(X_train[layer, :, :], y_train, X_test[layer, :, :], y_test, classifier_config)
        training_time = time() - start_time

        f1_train, roc_auc_train, pr_auc_train, acc_train, f1_test, roc_auc_test, pr_auc_test, acc_test, n_iter = sklearn_score(
            lr,
            X_train[layer, :, :],
            y_train,
            X_test[layer, :, :],
            y_test,
            binary=is_binary
        )
        if verbose:
            print(f'layer {layer + 1}:\tf1 train {f1_train:.4f},\taccuracy train {acc_train:.4f}\tf1 test {f1_test:.4f},\taccuracy test {acc_test:.4f},\ttime {training_time:.4f},\titers {n_iter}')

        scores['f1_train'].append(f1_train)
        scores['f1_test'].append(f1_test)
        scores['roc_auc_train'].append(roc_auc_train)
        scores['roc_auc_test'].append(roc_auc_test)
        scores['pr_auc_train'].append(pr_auc_train)
        scores['pr_auc_test'].append(pr_auc_test)
        scores['acc_train'].append(acc_train)
        scores['acc_test'].append(acc_test)
        scores['training_time'].append(training_time)
        scores['n_iter'].append(n_iter)
        scores['lrs'].append(lr)
    return scores


def run_experiment_from_pc(
        folder: str,
        classifier_func: Callable,
        classifier_config: dict,
        all_layers: bool = True,
        standartize_data: bool = False,
        verbose: bool = True,
        sample_idx: Optional[Iterable] = None
    ):
    """ loads vectors, trains classifier and returns scores """
    X_train = np.load('{}/X_train.npy'.format(folder))
    y_train = np.load('{}/y_train.npy'.format(folder))
    X_test = np.load('{}/X_test.npy'.format(folder))
    y_test = np.load('{}/y_test.npy'.format(folder))

    if standartize_data:
        X_train_std = []  # holder for standartized train data
        X_test_std = []  # holder for standartized test data
        for layer in range(X_train.shape[0]):
            # transforming vectors from each layer
            ss_i = StandardScaler(with_mean=True, with_std=True)
            X_train_i = ss_i.fit_transform(X_train[layer])
            X_test_i = ss_i.transform(X_test[layer])
            X_train_std.append(X_train_i)
            X_test_std.append(X_test_i)
        X_train = np.stack(X_train_std)
        X_test = np.stack(X_test_std)
        # free memory implicitly
        del X_train_std
        del X_test_std
        gc.collect()

    if sample_idx is not None:
        sample_idx = np.asarray(sample_idx)
        X_train = X_train[:, sample_idx]
        y_train = y_train[sample_idx]
    
    scores = experiment(classifier_func, classifier_config, X_train, y_train, X_test, y_test,
                        all_layers=all_layers, verbose=verbose)

    # free memory implicitly
    del X_train
    del y_train
    del X_test
    del y_test
    gc.collect()

    return scores


def show_exp(f1s: Iterable[float], accs: Iterable[float], title: str) -> Figure:
    """ Рисует графики скоров """
    fig = plt.figure()
    layers = range(len(f1s))
    plt.plot(layers, f1s, label='f1')
    plt.plot(layers, accs, label='accuracy')
    plt.title(title)
    plt.xlabel('Layer')
    plt.ylabel('Metric score')
    plt.legend()
    plt.show()
    return fig


if __name__ == '__main__':
    config = {
        'tol': 1e-3,
        'max_iter': 100,
        'solver': 'sag',
        'random_state': RANDOM_STATE,
        'verbose': True,
    }
    folder = 'TenseBert768_mean'
    scores = run_experiment_from_pc(folder, logreg, config, all_layers=False, standartize_data=True, verbose=False)
    json.dump(scores, sys.stdout)
