from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class AdditionalArguments:
    decoder_start_skim_layer: Optional[int] = field(default=1)
    gumbel_softmax_tau: Optional[float] = field(default=None)
    encoder_skim_factor: Optional[float] = field(default=None)
    decoder_skim_factor: Optional[float] = field(default=None)
    cross_skim_factor: Optional[float] = field(default=None)

def update_autoconfig(config, additional_args, **kwargs):
    s2s_config = {
        "gumbel_softmax_tau": additional_args.gumbel_softmax_tau,
        "encoder_skim_factor": additional_args.encoder_skim_factor,
        "decoder_skim_factor": additional_args.decoder_skim_factor,
        "cross_skim_factor": additional_args.cross_skim_factor,
        "decoder_start_skim_layer": additional_args.decoder_start_skim_layer,
    }
    config.update(s2s_config)

    return config
