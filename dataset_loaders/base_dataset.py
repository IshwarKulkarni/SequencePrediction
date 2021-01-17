import logging
from typing import List

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def collate_batch_fn(batch):
    xs = [i[0] for i in batch]
    ys = [i[1] for i in batch]
    return tuple([torch.stack(xs), torch.stack(ys)])


class TrainableDataset(Dataset):
    def __init__(self, dataset_in_data, dataset_out_data, input_seq_len, output_seq_len, overlap):
        super().__init__()
        self._in_data = dataset_in_data
        self._out_data = dataset_out_data

        self._input_scale = self._in_data.mean(0) / 25
        self._output_scale = self._out_data.mean(0) / 25

        self._in_data /= self._input_scale
        self._out_data /= self._output_scale

        self._input_seq_len = input_seq_len
        self._output_seq_len = output_seq_len

        self._overlap = overlap

    def __len__(self):
        sample_len = self._input_seq_len + self._output_seq_len
        data_len = self._in_data.shape[0]
        if self._overlap:
            return data_len - sample_len + 1
        return int(data_len / sample_len)

    def __getitem__(self, index):
        if not self._overlap:
            index = index * (self._input_seq_len + self._output_seq_len)
        end_1 = index + self._input_seq_len
        end_2 = end_1 + self._output_seq_len
        ip, op = self._in_data[index:end_1], self._out_data[index:end_2]
        return (torch.from_numpy(ip).float(), torch.from_numpy(op).float())

    def scale(self, ip, op):
        if ip is not None:
            ip *= self._input_scale
        if op is not None:
            op *= self._output_scale
        return ip, op


class DatasetMixer(torch.utils.data.Dataset):
    def __init__(self, datasets: List[TrainableDataset]):

        self.datasets = []
        self.tot_len = 0
        for d in datasets:
            self.datasets.append((self.tot_len, d))
            self.tot_len += len(d)
        self.datasets.reverse()
        self._name = '-'.join([d.name for d in datasets])
        logging.info(f'Dataset combining {len(datasets)} datasets resulting in {self.tot_len} total rows')

        self.name = '-'.join(d.name.upper() for d in datasets)

    def __getitem__(self, index):
        for offset, dataset in self.datasets:
            if index + 1 > offset:
                return dataset[index - offset]
        raise ValueError(f'Index offset {index} out of bounds')

    def __len__(self):
        return self.tot_len
