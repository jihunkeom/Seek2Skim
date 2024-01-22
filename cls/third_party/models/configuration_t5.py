"""T5 model congfiguration"""

from transformers.configuration_t5 import T5Config


class T5Config(T5Config):
    def __init__(
            self,
            gumbel_softmax_tau=0.1,
            encoder_skim_factor=None,
            decoder_skim_factor=None,
            cross_skim_factor=None
            **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.gumbel_softmax_tau = gumbel_softmax_tau
        self.encoder_skim_factor = encoder_skim_factor
        self.decoder_skim_factor = decoder_skim_factor
        self.cross_skim_factor = cross_skim_factor
