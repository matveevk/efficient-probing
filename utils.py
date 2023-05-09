import numpy as np
import random
import torch
from torch.utils.data import DataLoader 
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from typing import List, Tuple, Type


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


def retrieve_embeddings_and_labels_bert(
        dataloader: DataLoader,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        sentence_aggregation: str = 'mean',
        device: str = DEVICE
    ) -> Tuple[List, List]:
    """ Converts batches of sentences to batches of embeddings from a BERT-like model """
    model = model.to(device).eval()
    for texts, labels in dataloader:
        with torch.no_grad():
            inputs = tokenizer(texts, add_special_tokens=True, padding='longest', return_tensors='pt').to(device)
            outputs = model(**inputs)
            if sentence_aggregation == 'sum':
                layer_embs = [layer_emb.sum(axis=1).squeeze().cpu().numpy() for layer_emb in outputs.hidden_states]  # for sum of embeddings
            elif sentence_aggregation == 'cls':
                layer_embs = [layer_emb[:, 0, :].cpu().numpy() for layer_emb in outputs.hidden_states]  # for cls token embedding
            elif sentence_aggregation == 'mean':
                layer_embs = []
                for layer_emb in outputs.hidden_states:
                    mask = inputs.attention_mask.unsqueeze(-1)  # to correctly divide the sum
                    aggregated = torch.sum(layer_emb * mask, axis=1) / torch.clamp(mask.sum(axis=1), min=EPS)
                    layer_embs.append(aggregated.cpu().numpy())
            else:
                raise NotImplementedError('not implemented sentence_aggregation {} in retrieve_embeddings_and_labels_bert'.format(sentence_aggregation))
            yield layer_embs, labels.tolist()
