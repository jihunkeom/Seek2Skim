from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class AdditionalArguments:
    exit_conf: Optional[str] = field(default=None, metadata={"help": "exit cirterion"})
    exit_position_temp: Optional[int] = field(default=None, metadata={"help": "exit cirterion"})
    calm_max_answer_length: Optional[int] = field(default=128, metadata={"help": "exit cirterion"})
    exit_min_layer: Optional[int] = field(default=None, metadata={"help": "exit cirterion"})
    patience_list: Optional[List[int]] = field(default=None, metadata={"help": "PABEE decoder patience"})
    entropy_list: Optional[List[float]] = field(default=None, metadata={"help": "DeeBERT entropy list"})
    confidence_list: Optional[List[float]] = field(default=None, metadata={"help": "CALM confidence list"})

    encoder_first_N: Optional[int] = field(default=None)
    decoder_first_N: Optional[int] = field(default=None)

    decoder_start_skim_layer: Optional[int] = field(default=1)

    gumbel_softmax_tau: Optional[float] = field(default=None)
    encoder_skim_factor: Optional[float] = field(default=None)
    decoder_skim_factor: Optional[float] = field(default=None)
    cross_skim_factor: Optional[float] = field(default=None)

def update_autoconfig(config, additional_args, **kwargs):
    first_n_config = {
        "encoder_first_N": additional_args.encoder_first_N,
        "decoder_first_N": additional_args.decoder_first_N,
        "decoder_start_skim_layer": additional_args.decoder_start_skim_layer,
    }
    config.update(first_n_config)

    ee_config = {
        "exit_conf": additional_args.exit_conf,
        "exit_position_temp": additional_args.exit_position_temp,
        "calm_max_answer_length": additional_args.calm_max_answer_length,
        "exit_min_layer": additional_args.exit_min_layer,
    }
    config.update(ee_config)

    s2s_config = {
        "gumbel_softmax_tau": additional_args.gumbel_softmax_tau,
        "encoder_skim_factor": additional_args.encoder_skim_factor,
        "decoder_skim_factor": additional_args.decoder_skim_factor,
        "cross_skim_factor": additional_args.cross_skim_factor,
    }
    config.update(s2s_config)

    return config