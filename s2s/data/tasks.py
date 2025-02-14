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
        
class Squad(AbstractTaskDataset):
    name = "squad"
    metric = [metrics.f1_score_with_invalid]
    task_specific_config = {"max_length": 5, "num_beams": 1}
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    
    def load_dataset(self, split):
        return datasets.load_dataset(self.name, split=split)
    
    def preprocessor(self, example, add_prefix=True):
        # answer = pad_punctuation(example["answers"]["text"][0])
        # question = pad_punctuation(example["question"])
        # context = pad_punctuation(example["context"])
        answer, question, context = example['answers']['text'][0], example['question'], example['context']
        source = ["question:", question, "context:", context]
        target = [answer]
        return self.seq2seq_format(source, target, add_prefix)
        
        
class IWSLT2017RONL(AbstractTaskDataset):
    name = 'iwslt2017-ro-nl'
    task_specific_config = {"max_length": 300, "num_beams": 1}
    pair = f"ro-nl"
    metrics = [metrics.bleu]
    
    def load_dataset(self, split):
        return datasets.load_dataset("iwslt2017", "iwslt2017-ro-nl", split=split)
    
    def preprocessor(self, example, add_prefix=True):
        src_texts = [example['translation']['ro']]
        tgt_texts = [example['translation']['nl']]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix=add_prefix,
                                   prefix="translate Romanian to Dutch")
        
class IWSLT2017ENNL(AbstractTaskDataset):
    name = 'iwslt2017-en-nl'
    task_specific_config = {"max_length": 300, "num_beams": 1}
    pair = f"en-nl"
    metrics = [metrics.bleu]
    
    def load_dataset(self, split):
        return datasets.load_dataset("iwslt2017", "iwslt2017-en-nl", split=split)
    
    def preprocessor(self, example, add_prefix=True):
        src_texts = [example['translation']['en']]
        tgt_texts = [example['translation']['nl']]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix=add_prefix,
                                   prefix="translate English to Dutch")
        
class IWSLT2017NLEN(AbstractTaskDataset):
    name = 'iwslt2017-nl-en'
    task_specific_config = {"max_length": 300, "num_beams": 1}
    pair = f"nl-en"
    metrics = [metrics.bleu]
    
    def load_dataset(self, split):
        return datasets.load_dataset("iwslt2017", "iwslt2017-en-nl", split=split)
    
    def preprocessor(self, example, add_prefix=True):
        src_texts = [example['translation']['nl']]
        tgt_texts = [example['translation']['en']]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix=add_prefix,
                                   prefix="translate Dutch to English")
        
class WMT14DEENTaskDataset(AbstractTaskDataset):
    name = "wmt14-de-en"
    task_specific_config = {'max_length': 300, 'num_beams': 1}
    pair = f"de-en"
    metrics = [metrics.bleu]

    def load_dataset(self, split):
        return datasets.load_dataset("wmt14", self.pair, split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = [example['translation']["de"]]
        tgt_texts = [example['translation']["en"]]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   prefix="translate German to English")

class WMT14ENDETaskDataset(AbstractTaskDataset):
    name = "wmt14-en-de"
    task_specific_config = {'max_length': 300, 'num_beams': 1}
    pair = f"de-en"
    metrics = [metrics.bleu]

    def load_dataset(self, split):
        return datasets.load_dataset("wmt14", self.pair, split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = [example['translation']["en"]]
        tgt_texts = [example['translation']["de"]]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   prefix="translate English to German")
        
class WMT14FRENTaskDataset(AbstractTaskDataset):
    name = "wmt14-fr-en"
    task_specific_config = {'max_length': 300, 'num_beams': 1}
    pair = f"fr-en"
    metrics = [metrics.bleu]

    def load_dataset(self, split):
        return datasets.load_dataset("wmt14", self.pair, split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = [example['translation']["fr"]]
        tgt_texts = [example['translation']["en"]]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   prefix="translate French to English")

class WMT14ENFRTaskDataset(AbstractTaskDataset):
    name = "wmt14-en-fr"
    task_specific_config = {'max_length': 300, 'num_beams': 1}
    pair = f"rr-en"
    metrics = [metrics.bleu]

    def load_dataset(self, split):
        return datasets.load_dataset("wmt14", self.pair, split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = [example['translation']["en"]]
        tgt_texts = [example['translation']["fr"]]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   prefix="translate English to French")

class WMT16ENROTaskDataset(AbstractTaskDataset):
    name = "wmt16-en-ro"
    task_specific_config = {'max_length': 300, 'num_beams': 1}
    pair = f"ro-en"
    metrics = [metrics.bleu]

    def load_dataset(self, split):
        return datasets.load_dataset("wmt16", self.pair, split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = [example['translation']["en"]]
        tgt_texts = [example['translation']["ro"]]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   prefix="translate English to Romanian")


class WMT16ROENTaskDataset(AbstractTaskDataset):
    name = "wmt16-ro-en"
    task_specific_config = {'max_length': 300, 'num_beams': 1}
    pair = f"ro-en"
    metrics = [metrics.bleu]

    def load_dataset(self, split):
        return datasets.load_dataset("wmt16", self.pair, split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = [example['translation']["ro"]]
        tgt_texts = [example['translation']["en"]]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   prefix="translate Romanian to English")


class WMT16ENCSTaskDataset(AbstractTaskDataset):
    name = "wmt16-en-cs"
    task_specific_config = {'max_length': 300, 'num_beams': 1}
    pair = f"cs-en"
    metrics = [metrics.bleu]

    def load_dataset(self, split):
        return datasets.load_dataset("wmt16", self.pair, split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = [example['translation']["en"]]
        tgt_texts = [example['translation']["cs"]]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   prefix="translate English to Czech")
        
class WMT16CSENTaskDataset(AbstractTaskDataset):
    name = "wmt16-cs-en"
    task_specific_config = {'max_length': 300, 'num_beams': 1}
    pair = f"cs-en"
    metrics = [metrics.bleu]

    def load_dataset(self, split):
        return datasets.load_dataset("wmt16", self.pair, split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = [example['translation']["cs"]]
        tgt_texts = [example['translation']["en"]]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   prefix="translate Czech to English")


class WMT16ENFITaskDataset(AbstractTaskDataset):
    name = "wmt16-en-fi"
    task_specific_config = {'max_length': 300, 'num_beams': 1}
    pair = f"fi-en"
    metrics = [metrics.bleu]

    def load_dataset(self, split):
        return datasets.load_dataset("wmt16", self.pair, split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = [example['translation']["en"]]
        tgt_texts = [example['translation']["fi"]]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   prefix="translate English to Finnish")

class WMT16FIENTaskDataset(AbstractTaskDataset):
    name = "wmt16-fi-en"
    task_specific_config = {'max_length': 300, 'num_beams': 1}
    pair = f"fi-en"
    metrics = [metrics.bleu]

    def load_dataset(self, split):
        return datasets.load_dataset("wmt16", self.pair, split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = [example['translation']["fi"]]
        tgt_texts = [example['translation']["en"]]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   prefix="translate Finnish to English")

class WMT14HIENTaskDataset(AbstractTaskDataset):
    name = "wmt14-hi-en"
    task_specific_config = {'max_length': 300, 'num_beams': 1}
    pair = f"hi-en"
    metrics = [metrics.bleu]

    def load_dataset(self, split):
        return datasets.load_dataset("wmt14", self.pair, split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = [example['translation']["en"]]
        tgt_texts = [example['translation']["hi"]]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   prefix="translate English to Hindi")

class SamSum(AbstractTaskDataset):
    name = "samsum"
    task_specific_config = {"max_length": 200, "min_length": 30, "no_repeat_ngram_size": 3, "num_beams": 1}
    metrics = [metrics.rouge]
    
    def load_dataset(self, split):
        return datasets.load_dataset("samsum", split=split)
    
    def preprocessor(self, example, add_prefix=True):
        src_texts = [example["dialogue"]]
        tgt_texts = [example["summary"]]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   prefix="summarize: ")

class XSum(AbstractTaskDataset):
    name = "xsum"
    task_specific_config = {"max_length": 200, "min_length": 30, "no_repeat_ngram_size": 3, "num_beams": 1}
    metrics = [metrics.rouge]
    
    def load_dataset(self, split):
        return datasets.load_dataset("EdinburghNLP/xsum", split=split)
    
    def preprocessor(self, example, add_prefix=True):
        src_texts = [example["document"]]
        tgt_texts = [example["summary"]]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   prefix="summarize: ")
    
class CnnDailyMail(AbstractTaskDataset):
    name = "cnn_dailymail"
    task_specific_config = {"max_length": 200, "min_length": 30, "no_repeat_ngram_size": 3, "num_beams": 1}
    metrics = [metrics.rouge]
    
    def load_dataset(self, split):
        return datasets.load_dataset("cnn_dailymail", "3.0.0", split=split)
    
    def preprocessor(self, example, add_prefix=True):
        src_texts = [example["article"]]
        tgt_texts = [example["highlights"]]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   prefix="summarize: ")
        
class MultiNews(AbstractTaskDataset):
    name = "multi_news"
    task_specific_config = {"max_length": 200, "min_length": 30, "no_repeat_ngram_size": 3, "num_beams": 1}
    metrics = [metrics.rouge]
    
    def load_dataset(self, split):
        return datasets.load_dataset("multi_news", split=split)
    
    def preprocessor(self, example, add_prefix=True):
        src_texts = [example["document"]]
        tgt_texts = [example["summary"]]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   prefix="summarize: ")
    
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
    
class COLATaskDataset(AbstractTaskDataset):
    name = "cola"
    label_list = ["0", "1"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    metrics = [metrics.matthews_corrcoef]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'cola', split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence:", example['sentence']]
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


class STSBTaskDataset(AbstractTaskDataset):
    name = "stsb"
    label_list = [str(np.round(label, decimals=1)) for label in np.arange(0, 5.2, 0.2)]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    metrics = [metrics.pearson_corrcoef, metrics.spearman_corrcoef]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'stsb', split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence1:", example['sentence1'],
                     "sentence2:", example["sentence2"]]
        tgt_texts = [str(round_stsb_target(example['label']))]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class QQPTaskDataset(AbstractTaskDataset):
    name = "qqp"
    label_list = ["0", "1"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    metrics = [metrics.f1_score_with_invalid, metrics.accuracy]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'qqp', split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["question1:", example['question1'],
                     "question2:", example["question2"]]
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


class RTETaskDataset(AbstractTaskDataset):
    name = "rte"
    label_list = ["0", "1"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    metrics = [metrics.accuracy]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'rte', split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence1:", example['sentence1'],
                     "sentence2:", example["sentence2"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class WNLITaskDataset(AbstractTaskDataset):
    name = "wnli"
    label_list = ["0", "1"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    metrics = [metrics.accuracy]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'wnli', split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence1:", example['sentence1'],
                     "sentence2:", example["sentence2"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)
    
class SuperGLUEBoolQTaskDataset(AbstractTaskDataset):
    name = "superglue-boolq"
    label_list = ['0', '1']
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metrics = [metrics.accuracy]
    
    def load_dataset(self, split):
        return datasets.load_dataset("super_glue", 'boolq', split=split)
    
    def preprocessor(self, example, add_prefix=True):
        src_texts = ["question:", example["question"], "passage:", example["passage"]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)
    
class SuperGLUERTETaskDataset(AbstractTaskDataset):
    name = "superglue-rte"
    label_list = ['0', '1']
    task_specific_config = {"max_length": compute_task_max_decoding_length(label_list)}
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metrics = [metrics.accuracy]
    
    def load_dataset(self, split):
        return datasets.load_dataset('super_glue', "rte", split=split)
    
    def preprocessor(self, example, add_prefix=True):
        src_texts = ["premise", example["premise"], "hypothesis:", example["hypothesis"]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)
    
class SuperGLUECBTaskDataset(AbstractTaskDataset):
    name = "superglue-cb"
    label_list = ['0', '1', '2']
    task_specific_config = {"max_length": compute_task_max_decoding_length(label_list)}
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metrics = [metrics.accuracy]
    
    def load_dataset(self, split):
        return datasets.load_dataset('super_glue', 'cb', split=split)
    
    def preprocessor(self, example, add_prefix=True):
        src_texts = ["premise:", example['premise'], "hypothesis:", example["hypothesis"]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)
    
TASK_MAPPING = OrderedDict([
    ('superglue-boolq', SuperGLUEBoolQTaskDataset),
    ('superglue-cb', SuperGLUECBTaskDataset),
    ("superglue-rte", SuperGLUERTETaskDataset),
    ('mrpc', MRPCTaskDataset),
    ('cola', COLATaskDataset),
    ('sst2', SST2TaskDataset),
    ('stsb', STSBTaskDataset),
    ('qqp', QQPTaskDataset),
    ('mnli', MNLITaskDataset),
    ('qnli', QNLITaskDataset),
    ('rte', RTETaskDataset),
    ('wnli', WNLITaskDataset),
    ("wmt14-de-en", WMT14DEENTaskDataset),
    ("wmt14-en-de", WMT14ENDETaskDataset),
    ("wmt14-fr-en", WMT14FRENTaskDataset),
    ("wmt14-en-fr", WMT14ENFRTaskDataset),
    ("wmt16-ro-en", WMT16ROENTaskDataset),
    ('wmt14-hi-en', WMT14HIENTaskDataset),
    ('wmt16-en-ro', WMT16ENROTaskDataset),
    ('wmt16-en-cs', WMT16ENCSTaskDataset),
    ('wmt16-en-fi', WMT16ENFITaskDataset),
    ('wmt16-cs-en', WMT16CSENTaskDataset),
    ('wmt16-fi-en', WMT16FIENTaskDataset),
    ('iwslt2017-ro-nl', IWSLT2017RONL),
    ('iwslt2017-en-nl', IWSLT2017ENNL),
    ('iwslt2017-nl-en', IWSLT2017NLEN),
    ('xsum', XSum),
    ('cnn_dailymail', CnnDailyMail),
    ('multi_news', MultiNews),
    ("samsum", SamSum),
    ("squad", Squad)]
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