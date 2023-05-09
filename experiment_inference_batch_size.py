from .datasets import SampleDataset
from .utils import DEVICE, EPS, RANDOM_STATE, set_seed
import gc
import numpy as np
import pandas as pd
import random
import torch
from time import time
from tqdm.notebook import tqdm


# --- utility functions

def measure_inference_time_per_batch(model, texts, labels = None, device: str = 'cuda'):
    """ Seperate batch function for memory efficiency.
        :return: aggregation_time, forward_pass_time, tokenization_time
    """
    start = time()
    inputs = tokenizer(texts, add_special_tokens=True, padding='longest', return_tensors='pt').to(device)
    med_time = time()
    outputs = model(**inputs)
    med_time2 = time()
    layer_embs = []
    for layer_emb in outputs.hidden_states:
        mask = inputs.attention_mask.unsqueeze(-1)  # to correctly divide the sum
        aggregated = torch.sum(layer_emb * mask, axis=1) / torch.clamp(mask.sum(axis=1), min=EPS)
        layer_embs.append(aggregated)
    return time() - med_time2, med_time2 - med_time, med_time - start


def measure_inference_time(dataloader, model, tokenizer, device: str = DEVICE, verbose: bool = True):
    """ measures the efficiency of embedding retrieval (mean embedding) """
    tokenization_times = []
    forward_pass_times = []
    aggregation_times = []
    model = model.to(device).eval()
    with torch.no_grad():
        if verbose:
            dataloader = tqdm(dataloader)
        for texts, labels in dataloader:
            aggregation_time, forward_pass_time, tokenization_time = measure_inference_time_per_batch(model, texts, labels, device)
            forward_pass_times.append(forward_pass_time)
            tokenization_times.append(tokenization_time)
            aggregation_times.append(aggregation_time)
            torch.cuda.empty_cache()  # for RAM efficiency
    return tokenization_times, forward_pass_times, aggregation_times


# --- utility functions: for outputing measures per model

def get_adjusted_time(times, p=0.05):
    """ removes p%-outlaying batches and returns total adjusted time """
    times = np.asarray(times)
    min_time, max_time = np.quantile(times, q=(p / 2, 1 - p / 2))
    adjusted_times = times[(min_time <= times) & (times <= max_time)]
    return adjusted_times.mean() * times.shape[0]


def output_model_measures_per_batch_size(model_name):
    print(model_name, 'tokenization times:')
    for bs, times in tokenization_times[model_name].items():
        print(bs, np.sum(times), get_adjusted_time(times), get_adjusted_time(times, p=0.1))
    print()
    print(model_name, 'forward pass times:')
    for bs, times in forward_pass_times[model_name].items():
        print(bs, np.sum(times), get_adjusted_time(times), get_adjusted_time(times, p=0.1))


# --- experiment functions

def full_grid_tiny_bert(sample_dataset: Dataset):
    # get the model
    model, tokenizer = get_pretrained('google/bert_uncased_L-12_H-128_A-2')

    # holder for dataloaders for each batch_size
    batch2dataloader = {}
    for p in range(13):
        batch_sz = 2 ** p
        batch2dataloader[batch_sz] = DataLoader(
            sample_dataset,
            batch_size=batch_sz,
            shuffle=False
        )

    # holders for experiment results
    tokenization_times = {}
    forward_pass_times = {}
    aggregation_times = {}

    # experiment:
    for batch_sz, dataloader in batch2dataloader.items():

        # measure inference
        cur_tokenization_times, cur_forward_pass_times, cur_aggregation_times = measure_inference_time(dataloader, model, tokenizer, verbose=True)

        # log inference
        tokenization_times[batch_sz] = cur_tokenization_times
        forward_pass_times[batch_sz] = cur_forward_pass_times
        aggregation_times[batch_sz] = cur_aggregation_times

        # empty cuda cache
        torch.cuda.empty_cache()

    # free memory
    del model
    del tokenizer
    gc.collect()

    return tokenization_times, forward_pass_times, aggregation_times


def part_grid_all_models(sample_dataloader):
    # nvidia_smi.nvmlInit()

    model2name = {
        'BERT Tiny': 'google/bert_uncased_L-12_H-128_A-2',
        'BERT Multilingual': 'bert-base-multilingual-cased',
        'DistilBERT': 'distilbert-base-uncased',
        'GPT-2': 'gpt2',
        'MiniLM': 'sentence-transformers/all-MiniLM-L6-v2',
    }

    batch2dataloader = {
        4: {},
        8: {},
        16: {},
        32: {},
        64: {},
        128: {},
        256: {},
        512: {},
        1024: {},
        2048: {},
        # 4096: {},
    }

    for batch_sz in batch2dataloader:
        batch2dataloader[batch_sz] = DataLoader(
            sample_dataset,
            batch_size=batch_sz,
            shuffle=False
        )

    torch.cuda.empty_cache()

    tokenization_times = {}  # dictionary with tokenization of batches times per model
    forward_pass_times = {}  # dictionary with forward passing of batches times per model
    aggregation_times = {}
    # gpu_usage = {}  # dictionary with used GPU RAM bytes per model
    # current_gpu_ram_usage = nvidia_smi.nvmlDeviceGetMemoryInfo(handle).used

    for model_name_human, model_name in model2name.items():

        tokenization_times[model_name_human] = {}
        forward_pass_times[model_name_human] = {}
        aggregation_times[model_name_human] = {}
        -- gpu_usage[model_name_human] = {}

        for batch_sz in batch2dataloader:

            # get the model
            model, tokenizer = get_pretrained(model_name)

            # measure inference
            cur_tokenization_times, cur_forward_pass_times, cur_aggregation_times = measure_inference_time(batch2dataloader[batch_sz], model, tokenizer, verbose=True)

            # log inference
            tokenization_times[model_name_human][batch_sz] = cur_tokenization_times
            forward_pass_times[model_name_human][batch_sz] = cur_forward_pass_times
            aggregation_times[model_name_human][batch_sz] = cur_aggregation_times

            # new_gpu_ram_usage = nvidia_smi.nvmlDeviceGetMemoryInfo(handle).used
            # gpu_usage[model_name_human][batch_sz] = new_gpu_ram_usage - current_gpu_ram_usage
            # current_gpu_ram_usage = new_gpu_ram_usage

            # empty cuda cache
            torch.cuda.empty_cache()

            # free memory
            del model
            del tokenizer
            gc.collect()
        
    return tokenization_times, forward_pass_times, aggregation_times


if __name__ == '__main__':
    set_seed(RANDOM_STATE)
    
    # creating data
    sample_dataset = SampleDataset(
        filepaths=[
            'sentence_length.txt',
            'word_content.txt',
            'top_constituents.txt',
            'tree_depth.txt',
            'coordination_inversion.txt',
            'odd_man_out.txt',
            'past_present.txt'
        ],
        n_samples=14285,  # so that the total number is 100,000.
    )
    
    tokenization_times, forward_pass_times, aggregation_times = full_grid_tiny_bert(sample_dataset)
    # plot graphs
    
    tokenization_times, forward_pass_times, aggregation_times = part_grid_all_models(sample_dataset)
    # plot graphs
    
    
    
