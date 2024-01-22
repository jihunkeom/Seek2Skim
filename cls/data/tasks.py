"""Implements different tasks and define the processors to convert each dataset
to a sequence to sequence format."""
import re
import abc
import datasets
import functools
import logging
import numpy as np
import torch
from typing import Callable, Dict, Mapping, List
from collections import OrderedDict

from metrics import metrics
from .utils import round_stsb_target, compute_task_max_decoding_length

logger = logging.getLogger(__name__)

class AbstractTaskDataset(abc.ABC):
    """Defines the abstract class for all the tasks.
    name: the name of the task.
    task_specific_config: specifies the special configuration needs
        to be passed to encoder when decoding each task. Since different
        tasks, have different output space, the maximum decoding lenght
        varies based on the tasks.
    preporcessor: a processor to convert the given dataset to the Seq2Seq format.
    metrics: specifies the metrics to evaluate the task based on them.
    split_to_data_split: since no all the time, differnet splits of the
        datasets are available, we define a mapping from the wanted split
        to the existing dataset splits.
    small_datasets_without_all_splits: List of strings, defines the name
        of all low-resource tasks in which not all train/test/validation
        splits are available.
    large_data_without_all_splits: List of strings, defines the name of
        all high-resouce tasks in which no all train/test/validation
        splits are available.
    """
    name = NotImplemented
    task_specific_config: Dict = NotImplemented
    preprocessor: Callable = NotImplemented
    metrics: List[Callable] = NotImplemented
    split_to_data_split: Mapping[str, str] = \
        {"train": "train", "validation": "validation", "test": "test"}
        
    small_datasets_without_all_splits = ["cola", "wnli", "rte", "trec", "superglue-cb", "sick",
                                         "mrpc", "stsb", "imdb", "commonsense_qa", "superglue-boolq"]
    large_datasets_without_all_splits = ["yelp_polarity", "qqp", "qnli", "mnli",
                                         "social_i_qa", "cosmos_qa", "winogrande", "hellaswag", "sst2"]
    
    def __init__(self, seed=42):
        self.seed = seed
        
    def get_sampled_split(self, split, n_obs: int = None):
        # If the requested number of observation is more than dataset
        # size we reset it to the maximum available
        split = self.split_to_data_split[split]
        dataset = self.load_dataset(split)
        total_size = len(dataset)
        n_obs = self.check_n_obs(n_obs, total_size)
        if n_obs is not None:
            split = split + "[:{}]".format(n_obs)
        return split
    
    def get_shuffled_sampled_split(self, split, n_obs: int = None):
        # Defines the random generator.
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        # If the requested number of observation is more than dataset
        # size we reset it to the maximum available
        mapped_split = self.split_to_data_split[split]
        dataset = self.load_dataset(mapped_split)
        # shuffle the dataset and get the random samples.
        train_size = len(dataset)
        indices = torch.randperm(train_size, generator=generator).tolist()
        dataset = self.select_dataset_samples(indices, dataset, n_obs=n_obs)
        return dataset
        
    
    def check_n_obs(self, n_obs, total_size):
        if n_obs is not None and n_obs > total_size:
            n_obs = total_size
            logger.warning("n_obs is set to %s", n_obs)
        return n_obs
    
    def select_dataset_samples(self, indices, dataset, n_obs: int = None):
        n_obs = self.check_n_obs(n_obs, len(indices))
        indices = indices[:n_obs] if n_obs is not None else indices
        return dataset.select(indices)
    
    def load_dataset(self, split):
        return datasets.load_dataset(self.name, split=split)
    
    def get_train_split_indices(self, split):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        mapped_split = self.split_to_data_split["train"]
        dataset = self.load_dataset(mapped_split)
        train_size = len(dataset)
        indices = torch.randperm(train_size, generator=generator).tolist()
        validation_size = 1000
        if split == "validation":
            return indices[:validation_size]
        else:
            return indices[validation_size:]
    
    def get_half_validation_indices(self, split):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        mapped_split = self.split_to_data_split["validation"]
        dataset = self.load_dataset(mapped_split)
        validation_size = len(dataset)
        indices = torch.randperm(validation_size, generator=generator).tolist()
        if split == "validation":
            return indices[:validation_size//2]
        else:
            return indices[validation_size//2:]
    
    def get_dataset(self, split, n_obs=None, add_prefix=True, split_validation_test=False):
        if split_validation_test and self.name in self.small_datasets_without_all_splits and split != "train":
            mapped_split = self.split_to_data_split["validation"]
            dataset = self.load_dataset(split=mapped_split)
            indices = self.get_half_validation_indices(split)
            dataset = self.select_dataset_samples(indices, dataset, n_obs)
        elif split_validation_test and self.name in self.large_datasets_without_all_splits and split != "test":
            dataset = self.load_dataset(split="train")
            indices = self.get_train_split_indices(split)
            dataset = self.select_dataset_samples(indices, dataset, n_obs)
        else:
            if n_obs == -1:
                split = self.get_sampled_split(split, n_obs)
                dataset = self.load_dataset(split)
            else:
                dataset = self.get_shuffled_sampled_split(split, n_obs)
        
        return dataset.map(functools.partial(self.preprocessor, add_prefix=add_prefix),
                           remove_columns=dataset.column_names)
        
    def seq2seq_format(self, src_strs: List[str], tgt_strs: List[str],
                       add_prefix: bool = False, prefix: str = None):
        src_prefix = prefix if prefix is not None else self.name
        src_strs = [src_prefix] + src_strs if add_prefix else src_strs
        return {"src_texts": " ".join(src_strs),
                "tgt_texts": " ".join(tgt_strs),
                "task": self.name}
      
    
class MRPCTaskDataset(AbstractTaskDataset):
    name = "mrpc"
    label_list = ["0", "1"]
    task_specific_config = {"max_length": compute_task_max_decoding_length(label_list)}
    metrics = [metrics.accuracy, metrics.f1_score_with_invalid]
    split_to_data_split = {"train": "train", "validation": "validation", "test": "validation"}
    
    def load_dataset(self, split):
        return datasets.load_dataset('glue', "mrpc", split=split)
    
    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence1:", example["sentence1"],
                     "sentence2:", example["sentence2"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SST2TaskDataset(AbstractTaskDataset):
    name = "sst2"
    label_list = ["0", "1"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    metrics = [metrics.accuracy]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset("glue", "sst2", split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence:", example['sentence']]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class MNLITaskDataset(AbstractTaskDataset):
    name = "mnli"
    label_list = ["0", "1", "2"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    split_to_data_split = {"train": "train",
                           "validation": "validation_mismatched",
                           "test": "validation_matched"}
    metrics = [metrics.accuracy]

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'mnli', split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["premise:", example['premise'],
                     "hypothesis", example["hypothesis"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class QNLITaskDataset(AbstractTaskDataset):
    name = "qnli"
    label_list = ["0", "1"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    metrics = [metrics.accuracy]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'qnli', split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["question:", example['question'],
                     "sentence:", example["sentence"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)

    
TASK_MAPPING = OrderedDict([
    ('mrpc', MRPCTaskDataset),
    ('sst2', SST2TaskDataset),
    ('mnli', MNLITaskDataset),
    ('qnli', QNLITaskDataset)
)

class AutoTask:
    @classmethod
    def get(self, task_name, seed=42):
        if task_name in TASK_MAPPING:
            return TASK_MAPPING[task_name](seed)
        raise ValueError(
            f"Unrecognized task {task_name} for AutoTask Model"
            f"Task name should be one of {', '.join(c for c in TASK_MAPPING.keys())}."
        )
