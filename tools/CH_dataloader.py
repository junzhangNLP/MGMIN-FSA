import pickle
import numpy as np
import torch
# from torch._six import string_classes
int_classes = int
string_classes = str
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM
model_path = ""
tokenizer = AutoTokenizer.from_pretrained(model_path)
def bert_inputs(data):
    data = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs = data,
        truncation = True,
        padding='max_length',
        max_length=300,
        return_length=True,
        return_tensors='pt'
    )
    input_ids = data['input_ids']
    att_mask = data["attention_mask"]
    token_type_ids = data["token_type_ids"]
    return input_ids, att_mask,token_type_ids

class MMDataset(Dataset):
    def __init__(self, featurePath, mode='test'):
        self.mode = mode
        self.featurePath = featurePath
        self.__init_fgmsa()
    def __init_fgmsa(self):
        with open(self.featurePath, 'rb') as f:
            data = pickle.load(f)
        self.raw_test = list(data[self.mode]['raw_text'])
        self.input_ids, self.att_mask,self.token_type_ids = bert_inputs(self.raw_test)
        self.vision, self.audio = list(data[self.mode]['vision']), list(data[self.mode]['audio'])
        self.ids = data[self.mode]['id']
        self.labels = {
            'M': np.array(data[self.mode]['regression_labels']).astype(np.float32)
        }
        
        for m in "TAV":
            self.labels[m] = data[self.mode]['regression' + '_labels_' + m].astype(np.float32)


    def __len__(self):
        return len(self.labels['M'])

    def __getitem__(self, index):
        sample = {
            'input_ids': self.input_ids[index],
            'att_mask': self.att_mask[index], 
            "token_type_ids": self.token_type_ids[index],
            'audio': torch.Tensor(self.audio[index]),
            'vision': torch.Tensor(self.vision[index]),
            'index': index,
            'id': self.ids[index],
            'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()}
        }

        return sample

def MMDataLoader(featurePath):
    datasets = {
        'train': MMDataset(featurePath, mode='train'),
        'valid': MMDataset(featurePath, mode='valid'),
        'test': MMDataset(featurePath, mode='test')
    }

    dataLoader = {
        ds: DataLoader(datasets[ds],
                       batch_size=4,
                       shuffle=False
                       )
        for ds in datasets.keys()
    }

    return dataLoader
