import gc
import numpy as np
import random
import sys
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader 
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from typing import Callable, Iterable, List, Optional, Tuple, Type


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPS = 1e-8
RANDOM_STATE = 3407


def set_seed(random_state: int = RANDOM_STATE) -> None:
    torch.manual_seed(random_state)
    random.seed(random_state)
    np.random.seed(random_state)


def get_pretrained(
        model_name: str,
        model_class: Type[PreTrainedModel] = AutoModel,
        tokenizer_class: Type[PreTrainedTokenizer] = AutoTokenizer,
        device: str = DEVICE
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """ Returns huggingface model and tokenizer for model_name """
    tokenizer = tokenizer_class.from_pretrained(model_name, use_fast=True)
    if 'gpt' in model_name:
        tokenizer.pad_token = tokenizer.eos_token
    model = model_class.from_pretrained(model_name, output_hidden_states=True)
    model = model.to(device).eval()
    return model, tokenizer


def retrieve_embeddings_and_labels(
        dataloader: DataLoader,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        sentence_aggregation: str = 'mean',
        device: str = DEVICE
    ) -> Tuple[List, List]:
    """ Converts batches of sentences to batches of embeddings """
    model = model.to(device).eval()
    for texts, labels in dataloader:
        with torch.no_grad():
            inputs = tokenizer(texts, add_special_tokens=True, padding='longest', return_tensors='pt').to(device)
            outputs = model(**inputs)
            if sentence_aggregation == 'sum':
                layer_embs = [layer_emb.sum(axis=1).squeeze() for layer_emb in outputs.hidden_states]  # for sum of embeddings
            elif sentence_aggregation == 'cls':
                layer_embs = [layer_emb[:, 0, :] for layer_emb in outputs.hidden_states]  # for cls token embedding
            elif sentence_aggregation == 'mean':
                layer_embs = []
                for layer_emb in outputs.hidden_states:
                    mask = inputs.attention_mask.unsqueeze(-1)  # to correctly divide the sum
                    aggregated = torch.sum(layer_emb * mask, axis=1) / torch.clamp(mask.sum(axis=1), min=EPS)
                    layer_embs.append(aggregated)
            elif sentence_aggregation == 'max':
                layer_embs = []
                for layer_emb in outputs.hidden_states:
                    mask = inputs.attention_mask.unsqueeze(-1).expand(layer_emb.size())  # to correctly divide the sum
                    layer_emb[mask == 0] = -INF
                    aggregated = torch.max(layer_emb, 1)[0]
                    layer_embs.append(aggregated)
            elif sentence_aggregation == 'eos':
                layer_embs = [layer_emb[:, -1, :] for layer_emb in outputs.hidden_states]  # for eos token embedding
            else:
                raise NotImplementedError('not implemented sentence_aggregation {} in retrieve_embeddings_and_labels_bert'.format(sentence_aggregation))
            yield layer_embs, labels.tolist()


def retrieve_embeddings_and_labels_bert(*args, **kwargs) -> Tuple[List, List]:
    """ [Redundant] Same as retrieve_embeddings_and_labels """
    return retrieve_embeddings_and_labels(*args, **kwargs)


def build_data_tensor(dataloader: DataLoader, **retriever_params) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Runs dataloader through retriever function.
        :param dataloader: dataloader from ProbingDataset
        :param retriever_params: params passed to retrieve_embeddings_and_labels_bert
        :return: torch.tensor with sentence embeddings, of shape <n_layers, n_samples, n_dim>, and torch.tensor of labels
    """
    stack_embs = []
    stack_labels = []
    for embs, labels in tqdm(retrieve_embeddings_and_labels(dataloader, **retriever_params)):
        stack_embs.append(torch.stack(embs))
        stack_labels += labels
    X = torch.concatenate(stack_embs, axis=1)
    y = torch.tensor(stack_labels)
    return X, y


def build_data_ndarray(dataloader: DataLoader, **retriever_params) -> Tuple[np.ndarray, np.array]:
    """ Runs dataloader through retriever function.
        :param dataloader: dataloader from ProbingDataset
        :param retriever_params: params passed to retrieve_embeddings_and_labels_bert
        :return: np.ndarray with sentence embeddings, of shape <n_layers, n_samples, n_dim>, and np.array of labels
    """
    X, y = build_data_tensor(dataloader, **retriever_params)
    return X.cpu().numpy(), y.cpu().numpy()


def standardize_data(X_train: np.ndarray, X_test: np.ndarray, X_val: Optional[np.ndarray] = None) -> Tuple[np.ndarray, ...]:
    """ standartizes input data with respect to X_train """
    X_train_std = []  # holder for standartized train data
    X_test_std = []  # holder for standartized test data
    if X_val is not None:
        X_val_std = []  # holder for standartized test data

    for layer in range(X_train.shape[0]):
        # transforming vectors from each layer
        ss_i = StandardScaler(with_mean=True, with_std=True)
        
        X_train_i = ss_i.fit_transform(X_train[layer])
        X_train_std.append(X_train_i)
        
        X_test_i = ss_i.transform(X_test[layer])
        X_test_std.append(X_test_i)

        if X_val is not None:
            X_val_i = ss_i.transform(X_val[layer])
            X_val_std.append(X_val_i)

    # stack and free memory explicitly
    X_train = np.stack(X_train_std)
    del X_train_std
    X_test = np.stack(X_test_std)
    del X_test_std
    if X_val is not None:
        X_val = np.stack(X_val_std)
        del X_val_std
        gc.collect()
        return X_train, X_test, X_val
    gc.collect()
    return X_train, X_test


def run_experiment_from_pc(
        folder: str,
        classifier_func: Callable,
        classifier_config: dict,
        all_layers: bool = True,
        standartize_data: bool = False,
        verbose: bool = True,
        sample_idx: Optional[Iterable] = None,
        feature_list: Optional[Iterable] = None,  # which features to use
        target_dim: Optional[int] = None,
    ) -> Dict[]:
    """ loads vectors, trains classifier and returns scores """
    X_train = np.load('{}/X_train.npy'.format(folder))
    y_train = np.load('{}/y_train.npy'.format(folder))
    X_test = np.load('{}/X_test.npy'.format(folder))
    y_test = np.load('{}/y_test.npy'.format(folder))

    # reducing dimension
    if feature_list is not None:
        X_train = X_train[:, :, feature_list]
        X_test = X_test[:, :, feature_list]

    # standartizing data
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
    
    # reducing dimension (deprecated)
    if target_dim is not None:
        raise ValueError('Reducing dimension automaticaly is deprecated. Please pass the indicies explicitly')
        X_train_pca = []  # holder for dim-reduced train data
        X_test_pca = []  # holder for dim-reduced test data
        for layer in range(X_train.shape[0]):
            # transforming vectors from each layer
            if all_layers or layer == X_train.shape[0] - 1:
                pca = PCA(n_components=target_dim)
                X_train_i = pca.fit_transform(X_train[layer])
                X_test_i = pca.transform(X_test[layer])
                X_train_pca.append(X_train_i)
                X_test_pca.append(X_test_i)
        X_train = np.stack(X_train_pca)
        X_test = np.stack(X_test_pca)
        # free memory explicitly
        del X_train_pca
        del X_test_pca
        gc.collect()

    # sampling data
    if sample_idx is not None:
        sample_idx = np.asarray(sample_idx)
        X_train = X_train[:, sample_idx]
        y_train = y_train[sample_idx]
    
    scores = experiment(classifier_func, classifier_config, X_train, y_train, X_test, y_test,
                        all_layers=all_layers, verbose=verbose)

    # free memory explicitly
    del X_train
    del y_train
    del X_test
    del y_test
    gc.collect()

    return scores
