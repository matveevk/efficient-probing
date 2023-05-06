import torch
import random
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel, DistilBertTokenizer, DistilBertModel
from tqdm.notebook import tqdm
from time import time


if __name__ == '__main__':

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # todo: move following constants to parameters
    RANDOM_STATE = 3407
    DATALOADER_BS = 128
    INF = 1e9
    CLASSIFIER_CONFIG = {
        'tol': 1e-2,
        'max_iter': 100,
        'solver': 'saga',
        'random_state': RANDOM_STATE,
        'multi_class': 'multinomial',
    }
    MODEL_NAME = 'google/bert_uncased_L-12_H-128_A-2'
    AGGR = 'sum'
    SAVE_DATA = False
    VERBOSE = True
    SUBSET_SZ = 100  # number of words to train on: used shrink the dataset
    if SUBSET_SZ is None:
        SUBSET_SZ = INF


    # setting random seed for reproducibility
    set_seed(RANDOM_STATE)

    # loading datasets
    # dataset file can be retrieved from SentEval using this command:
    # wget -c https://raw.githubusercontent.com/facebookresearch/SentEval/main/data/probing/word_content.txt
    word_content_train = WordContentDataset(
        'word_content.txt',
        subset='tr',
        shuffle=True
    )
    word_content_val = WordContentDataset(
        'word_content.txt',
        subset='va',
        shuffle=True,
        word2encoding=word_content_train.word2encoding
    )
    word_content_test = WordContentDataset(
        'word_content.txt',
        subset='te',
        shuffle=True,
        word2encoding=word_content_train.word2encoding
    )
    
    # making dataloaders from datasets
    word_content_train_loader = DataLoader(
        word_content_train,
        batch_size=DATALOADER_BS,
        shuffle=False
    )
    word_content_val_loader = DataLoader(
        word_content_val,
        batch_size=DATALOADER_BS,
        shuffle=False
    )
    word_content_test_loader = DataLoader(
        word_content_test,
        batch_size=DATALOADER_BS,
        shuffle=False
    )

    # obtaining model from huggingface
    model, tokenizer = get_pretrained(MODEL_NAME)

    # vectorizing X_train, y_train, X_val, y_val, X_test, y_test format
    train_stack_embs = []
    train_stack_labels = []
    embeddings_retriever = retrieve_embeddings_and_labels_bert(word_content_train_loader, sentence_aggregation=AGGR)
    if VERBOSE:
        embeddings_retriever = tqdm(embeddings_retriever, total=len(embeddings_retriever), desc='Retrieving embeddings on {}'.format('train'))
    for embs, labels in embeddings_retriever:
        train_stack_embs.append(np.stack(embs))
        train_stack_labels += labels
    X_train = np.concatenate(train_stack_embs, axis=1)
    y_train = np.asarray(train_stack_labels)
    del train_stack_embs
    del train_stack_labels
    
    test_stack_embs = []
    test_stack_labels = []
    embeddings_retriever = retrieve_embeddings_and_labels_bert(word_content_test_loader, sentence_aggregation=AGGR)
    if VERBOSE:
        embeddings_retriever = tqdm(embeddings_retriever, total=len(embeddings_retriever), desc='Retrieving embeddings on {}'.format('test'))
    for embs, labels in embeddings_retriever:
        test_stack_embs.append(np.stack(embs))
        test_stack_labels += labels
    X_test = np.concatenate(test_stack_embs, axis=1)
    y_test = np.asarray(test_stack_labels)
    del test_stack_embs
    del test_stack_labels

    val_stack_embs = []
    val_stack_labels = []
    embeddings_retriever = retrieve_embeddings_and_labels_bert(word_content_val_loader, sentence_aggregation=AGGR)
    if VERBOSE:
        embeddings_retriever = tqdm(embeddings_retriever, total=len(embeddings_retriever), desc='Retrieving embeddings on {}'.format('val'))
    for embs, labels in embeddings_retriever:
        val_stack_embs.append(np.stack(embs))
        val_stack_labels += labels
    X_val = np.concatenate(val_stack_embs, axis=1)
    y_val = np.asarray(val_stack_labels)
    del val_stack_embs
    del val_stack_labels
    
    if SAVE_DATA:
        np.save('X_train.npy', X_train)
        np.save('X_val.npy', X_val)
        np.save('X_test.npy', X_test)
        np.save('y_train.npy', y_train)
        np.save('y_val.npy', y_val)
        np.save('y_test.npy', y_test)
      
    # running probing
    f1s_train, accs_train, f1s_test, accs_test, times, n_iters, lrs = experiment(
        classifier_func=logreg,
        classifier_config=CLASSIFIER_CONFIG,
        X_train=X_train[:, y_train < SUBSET_SZ, :],
        y_train=y_train[y_train < SUBSET_SZ],
        X_test=X_test[:, y_test < SUBSET_SZ, :],
        y_test=y_test[y_test < SUBSET_SZ],
        verbose=VERBOSE
    )
