import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import ArrayLike
from scipy.stats import pearsonr, spearmanr
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from time import time, sleep
from typing import Any, Callable, Dict, Type



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
        f1 train, acc train, f1 test, acc test, n_iter on train.
        Бинарность классификации задаётся параметром binary
    """
    average = 'binary' if binary else 'macro'
    y_train_pred = lr.predict(X_train)
    y_test_pred = lr.predict(X_test)
    return f1_score(y_train, y_train_pred, average=average), \
        accuracy_score(y_train, y_train_pred), \
        f1_score(y_test, y_test_pred, average=average), \
        accuracy_score(y_test, y_test_pred), \
        lr.n_iter_


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
    verbose: bool = True
):
    """ Runs one probing classification experiment with given parameters. Parameters for the classifier are set in the config dict """
    if verbose:
        print('Classifier: {}, {}'.format(classifier_func.__name__, classifier_config))
    layers_cnt = X_train.shape[0]
    is_binary = True if len(np.unique(y_train)) == 2 else False

    if verbose:
        print('probing...')
    layers = list(range(layers_cnt))
    f1s_train = []
    f1s_test = []
    accs_train = []
    accs_test = []
    learn_times = []
    n_iters = []
    lrs = []
    for layer in layers:
        start_time = time()
        lr = classifier_func(X_train[layer, :, :], y_train, X_test[layer, :, :], y_test, classifier_config)
        learn_time = time() - start_time

        f1_train, acc_train, f1_test, acc_test, n_iter = sklearn_score(
            lr,
            X_train[layer, :, :],
            y_train,
            X_test[layer, :, :],
            y_test,
            binary=is_binary
        )
        if verbose:
            print(f'layer {layer + 1}:\tf1 train {f1_train:.4f},\taccuracy train {acc_train:.4f}\tf1 test {f1_test:.4f},\taccuracy test {acc_test:.4f},\ttime {learn_time:.4f},\titers {n_iter}')

        f1s_train.append(f1_train)
        f1s_test.append(f1_test)
        accs_train.append(acc_train)
        accs_test.append(acc_test)
        learn_times.append(learn_time)
        n_iters.append(n_iter)
        lrs.append(lr)
    return f1s_train, f1s_test, accs_train, accs_test, learn_times, n_iters, lrs


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