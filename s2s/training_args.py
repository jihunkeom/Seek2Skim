"""Defines the arguments used for training and evaluation."""
import torch
import logging
from dataclasses import dataclass, field
from transformers import TrainingArguments
from transformers.optimization import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from typing import Optional, List

arg_to_scheduler = {
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    "constant": get_constant_schedule,
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "constant_w_warmup": get_constant_schedule_with_warmup,
}

logger = logging.getLogger(__name__)


@dataclass
class Seq2SeqTrainingArguments(TrainingArguments):
    """
    Contains different training parameters such as dropout, optimizers parameters, ... .
    """
    label_smoothing: Optional[float] = field(
        default=0.0, metadata={"help": "The label smoothing epsilon to apply (if not zero)."}
    )
    predict_with_generate: bool = field(
        default=True, metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
    )
    adafactor: bool = field(default=False, metadata={"help": "whether to use adafactor"})
    encoder_layerdrop: Optional[float] = field(
        default=None, metadata={"help": "Encoder layer dropout probability. Goes into model.config."}
    )
    decoder_layerdrop: Optional[float] = field(
        default=None, metadata={"help": "Decoder layer dropout probability. Goes into model.config."}
    )
    dropout: Optional[float] = field(default=None, metadata={"help": "Dropout probability. Goes into model.config."})
    attention_dropout: Optional[float] = field(
        default=None, metadata={"help": "Attention dropout probability. Goes into model.config."}
    )
    lr_scheduler: Optional[str] = field(
        default="linear",
        metadata={"help": f"Which lr scheduler to use. Selected in {sorted(arg_to_scheduler.keys())}"},
    )
    temperature: Optional[int] = field(default=1, metadata={"help": "Defines the temperature"
                                                                    "value for sampling across the multiple datasets."})
    do_test: bool = field(default=False, metadata={"help": "Whether to comptue evaluation metrics on the test sets."})
    eval_output_dir: Optional[str] = field(default=None, metadata={
        "help": "The output directory where the evaluation of the model and checkpoints during "
        "evaluation will be written. Would use the original output_dir if not specified."})
    generate_classifier_weights: Optional[bool] = field(default=False, metadata={
        "help": "If set, generates the weights of the classifier by using a hyper-network."})
    optimize_from_scratch: Optional[bool] = field(default=False,
                                                  metadata={"help": "If set, this does not load the optimizers from"
                                                                    "the given model path."})
    optimize_from_scratch_with_loading_model: Optional[bool] = field(default=False, metadata={
        "help": "If set, it loads the model still but optimize from scratch."})
    split_validation_test: Optional[bool] = field(default=False,
                                                  metadata={"help": "If set, for the datasets which do not"
                                                                    "have the test set, we use validation set as their"
                                                                    "test set and make a validation set from either"
                                                                    "splitting the validation set into half (for smaller"
                                                                    "than 10K samples datasets), or by using 1K examples"
                                                                    "from training set as validation set (for larger"
                                                                    " datasets)."})
    print_num_parameters: Optional[str] = field(default=False,
                                                metadata={
                                                    "help": "If specified, prints the total number of parameters."})
    compute_memory: Optional[bool] = field(default=False,
                                           metadata={"help": "If specified, measures the memory needed."})
    compute_time: Optional[bool] = field(default=False, metadata={"help": "If specified, measures the time needed."})

    @property
    def device(self) -> "torch.device":
        return torch.device("cuda:0")
    
    tie_word_embeddings: Optional[bool] = field(default=True, metadata={"help": "tie input and output embeddings."})
    
    exit_conf: Optional[str] = field(default=None, metadata={"help": "exit cirterion"})
    patience_list: Optional[List[int]] = field(default=None, metadata={"help": "PABEE decoder patience"})
    entropy_list: Optional[List[float]] = field(default=None, metadata={"help": "DeeBERT entropy list"})
    confidence_list: Optional[List[float]] = field(default=None, metadata={"help": "CALM confidence list"})
    exit_position_temp: Optional[int] = field(default=None, metadata={"help": "exit cirterion"})
    max_answer_length: Optional[int] = field(default=None, metadata={"help": "exit cirterion"})
    
    encoder_threshold_list: Optional[List[float]] = field(default=None, metadata={"help": "Mue enc threshold list"})
    decoder_threshold_list: Optional[List[float]] = field(default=None, metadata={"help": "Mue dec threshold list"})

    encoder_first_N: Optional[int] = field(default=None)
    decoder_first_N: Optional[int] = field(default=None)
    gumbel_softmax_tau: Optional[float] = field(default=None)
    encoder_skim_factor: Optional[float] = field(default=None)
    decoder_skim_factor: Optional[float] = field(default=None)
    cross_skim_factor: Optional[float] = field(default=None)

@dataclass
class ModelArguments:
    """
    Contains the arguments defining model, tokenizer, and config which we use for finetuning.
    Also, it defines which parameters of the model needs to be freezed during finetuning.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    not_load_t5_checkpoint: bool = field(default=False, metadata={"help": "whether to load the checkpoint."})
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    freeze_encoder: bool = field(default=False, metadata={"help": "Whether tp freeze the encoder."})
    freeze_embeds: bool = field(default=False, metadata={"help": "Whether  to freeze the embeddings."})
    freeze_model: bool = field(default=False, metadata={"help": "Whether  to freeze the model."})
    freeze_model_but_lm_head: bool = field(default=False, metadata={"help": "Whether to freeze the"
        "whole model and only keep the language model head as the training parameter."})
    unfreeze_lm_head: bool = field(default=False, metadata={"help": "Whether  to unfreeze the lm_head."})
    freeze_model_but_task_embeddings: bool = field(default=False,
                                                   metadata={"help": "freezes the whole model but task-embedding."})
    unfreeze_layer_norms: bool = field(default=False, metadata={"help": "unfreezes the layer norms."})
    unfreeze_model: bool = field(default=False, metadata={"help": "Whether  to unfreeze the model."})


@dataclass
class DataTrainingArguments:
    """
    Arguments related to data used for training and evaluation.
    """
    tasks: Optional[List[str]] = field(
        default="MRPC",
        metadata={"help": "Task name from the list of registered tasks."},
    )
    eval_tasks: Optional[List[str]] = field(
        default="MRPC",
        metadata={"help": "Evaluation task name from the list of registered tasks."},
    )
    adapters: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Defines a dictionary from adapters to the tasks."}
    )
    task_embeddings: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Defines a dictionary from tasks to the tasks embeddings."}
    )
    max_source_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    test_max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for test target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    n_train: Optional[int] = field(default=-1, metadata={"help": "# training examples. -1 means use all."})
    n_val: Optional[int] = field(default=-1, metadata={"help": "# validation examples. -1 means use all."})
    n_test: Optional[int] = field(default=-1, metadata={"help": "# test examples. -1 means use all."})
    eval_beams: Optional[int] = field(default=None, metadata={"help": "# num_beams to use for evaluation."})
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={"help": "If only pad tokens should be ignored. This assumes that `config.pad_token_id` is defined."},
    )
    data_seed: Optional[int] = field(default=42, metadata={"help": "The seed used to subsample the datasets."})

