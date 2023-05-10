import sklearn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Iterable, List, Optional, Tuple

RANDOM_STATE = 3407


class ProbingDataset(Dataset):
    """
    Dataset for probing task.
    See example file: https://github.com/facebookresearch/SentEval/blob/main/data/probing/tree_depth.txt
    """
    def __init__(
        self,
        filepath: str,
        sep: str = '\t',
        columns: Optional[Iterable[str]] = None,
        subset: Optional[str] = None,
        shuffle: bool = False,
        random_state: int = RANDOM_STATE,
    ):
        """
        :param columns: how to name the columns of pd
        :param subset: tr, va, te, or None for all
        """
        super().__init__()
        # self.dataset = pd.read_csv(filepath, sep=sep, header=None if columns else 0)  # this is buggy
        dataframe = {
            'subset': [],
            'label': [],
            'text': [],
        }
        word2encoding = {}
        with open(filepath, 'r') as f:
            for row in f:
                row_subset, row_label, row_text = map(str.strip, row.split('\t'))
                dataframe['subset'].append(row_subset)
                dataframe['label'].append(row_label)
                dataframe['text'].append(row_text)
        self.dataset = pd.DataFrame.from_dict(dataframe)

        if columns is not None:
            self.dataset.columns = columns
        if subset is not None:
            self.dataset = self.dataset[self.dataset.subset == subset]
        if shuffle:
            self.dataset = sklearn.utils.shuffle(self.dataset, random_state=random_state)
        self.dataset = self.dataset.reset_index(drop=True)

    def __len__(self) -> int:
        return self.dataset.shape[0]
    
    def __getitem__(self, idx: int):
        text = self.dataset.text[idx]
        label = int(self.dataset.label[idx])
        return text.strip(), label


class WordContentDataset(ProbingDataset):
      """
      Dataset for Word Content (WC)
      See example file: https://github.com/facebookresearch/SentEval/blob/main/data/probing/word_content.txt
      """
    def __init__(
        self,
        filepath: str,
        sep: str = '\t',
        columns: Optional[Iterable[str]] = None,
        subset: Optional[str] = None,
        shuffle: bool = False,
        random_state: int = RANDOM_STATE,
        word2encoding: Optional[Dict[str, int]] = None,  # encoding dictionary from train dataset
    ):
        """
        :param columns: how to name the columns of pd
        :param subset: tr, va, te, or None for all
        """
        super().__init__()
        # self.dataset = pd.read_csv(filepath, sep=sep, header=None if columns else 0)  # this is buggy
        dataframe = {
            'subset': [],
            'label': [],
            'text': [],
            'label_encoded': []
        }
        self.word2encoding = word2encoding or {}
        
        with open(filepath, 'r') as f:
            for row in f:
                row_subset, row_label, row_text = map(str.strip, row.split('\t'))
                dataframe['subset'].append(row_subset)
                dataframe['label'].append(row_label)
                dataframe['text'].append(row_text)
                if row_label not in self.word2encoding:
                    self.word2encoding[row_label] = len(self.word2encoding)
                dataframe['label_encoded'].append(self.word2encoding[row_label])
        self.dataset = pd.DataFrame.from_dict(dataframe)

        if columns is not None:
            self.dataset.columns = columns
        if subset is not None:
            self.dataset = self.dataset[self.dataset.subset == subset]
        if shuffle:
            self.dataset = sklearn.utils.shuffle(self.dataset, random_state=random_state)
        self.dataset = self.dataset.reset_index(drop=True)


class TenseDataset(ProbingDataset):
    """
        Dataset for Past Present (Tense)
        See example file: https://raw.githubusercontent.com/facebookresearch/SentEval/main/data/probing/past_present.txt
    """
    def __getitem__(self, idx: int):
        text = self.dataset.text[idx]
        label = 0 if self.dataset.label[idx] == 'PAST' else 1
        return text.strip(), label
    
    
class TreeDepthDataset(ProbingDataset):
    """
        Dataset for Tree Depth
        See example file: https://raw.githubusercontent.com/facebookresearch/SentEval/main/data/probing/tree_depth.txt
    """
    def __getitem__(self, idx: int):
        text = self.dataset.text[idx]
        label = int(self.dataset.label[idx]) - 5
        return text.strip(), label

    
class SentLenDataset(ProbingDataset):
    """
        Dataset for Sentence Length
        See example file: https://raw.githubusercontent.com/facebookresearch/SentEval/main/data/probing/sentence_length.txt
    """
    pass


class TopConstituentsDataset(ProbingDataset):
    """
        Dataset for TopConstituents (WC)
        See example file: https://github.com/facebookresearch/SentEval/blob/main/data/probing/top_constituents.txt
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        all_labels = [
            'ADVP_NP_VP_.',
            'CC_ADVP_NP_VP_.',
            'CC_NP_VP_.',
            'IN_NP_VP_.',
            'NP_ADVP_VP_.',
            'NP_NP_VP_.',
            'NP_PP_.',
            'NP_VP_.',
            'OTHER',
            'PP_NP_VP_.',
            'RB_NP_VP_.',
            'SBAR_NP_VP_.',
            'SBAR_VP_.',
            'S_CC_S_.',
            'S_NP_VP_.',
            'S_VP_.',
            'VBD_NP_VP_.',
            'VP_.',
            'WHADVP_SQ_.',
            'WHNP_SQ_.',
        ]
        self.label_mapping = dict(zip(all_labels, range(len(all_labels))))

    def __getitem__(self, idx: int):
        text = self.dataset.text[idx]
        label = self.label_mapping[self.dataset.label[idx]]
        return text.strip(), label


class OddManOutDataset(ProbingDataset):
    """
        Dataset for Odd-Man-Out (SOMO)
        See example file: https://github.com/facebookresearch/SentEval/blob/main/data/probing/odd_man_out.txt
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        all_labels = ['O', 'C']
        self.label_mapping = dict(zip(all_labels, range(len(all_labels))))

    def __getitem__(self, idx: int):
        text = self.dataset.text[idx]
        label = self.label_mapping[self.dataset.label[idx]]
        return text.strip(), label


class SampleDataset(Dataset):
    """
    Creates dataset from multiple probing sources, randomly sampling the same amount from each of them
    """
    def __init__(
        self,
        filepaths: Iterable[str],
        n_samples: int = 1000,
        sep: str = '\t',
        random_state: int = RANDOM_STATE,
    ):
        """
        :param columns: how to name the columns of pd
        :param subset: tr, va, te, or None for all
        """
        super().__init__()
        self.sample_sentences = []
        for filepath in filepaths:
            sentences = []
            with open(filepath, 'r') as f:
                for row in f:
                    row_subset, row_label, row_text = map(str.strip, row.split('\t'))
                    sentences.append(row_text)
            random.seed(random_state)
            sample_sentences = random.sample(sentences, n_samples)
            self.sample_sentences += sample_sentences

    def __len__(self) -> int:
        return len(self.sample_sentences)
    
    def __getitem__(self, idx: int) -> str:
        return self.sample_sentences[idx]


if __name__ == '__main__':
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
        print(word_content_test.dataset.head())
