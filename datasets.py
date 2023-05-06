import sklearn
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Tuple

RANDOM_STATE = 3407


class WordContentDataset(Dataset):
      """
      Dataset for Word Content (WC)
      See example file: https://github.com/facebookresearch/SentEval/blob/main/data/probing/word_content.txt
      """
    def __init__(
        self,
        filepath: str,
        sep: str = '\t',
        columns: Optional[List[str]] = None,
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
        for row in open(filepath, 'r'):
            row_subset, row_label, row_text = row.split('\t')
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

    def __len__(self) -> int:
        return self.dataset.shape[0]
    
    def __getitem__(self, idx: int) -> Tuple[str, int]:
        text = self.dataset.text[idx]
        label = int(self.dataset.label_encoded[idx])
        return text.strip(), label



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
