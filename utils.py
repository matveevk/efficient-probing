import numpy as np
import random
import torch
from transformers import BertModel, BertTokenizer, DistilBertModel, DistilBertTokenizer, PreTrainedModel, PreTrainedTokenizer
from typing import Tuple, Type


RANDOM_STATE = 3407
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def set_seed(random_state: int = RANDOM_STATE) -> None:
    torch.manual_seed(random_state)
    random.seed(random_state)
    np.random.seed(random_state)


def get_pretrained(
        model_name: str,
        model_class: Type[PreTrainedModel] = BertModel,
        tokenizer_class: Type[PreTrainedTokenizer] = BertTokenizer,
        device: str = DEVICE
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """ Returns huggingface model and tokenizer for model_name """
    tokenizer = tokenizer_class.from_pretrained(model_name)
    model = model_class.from_pretrained(model_name, output_hidden_states=True)
    model = model.to(device)
    return model, tokenizer
