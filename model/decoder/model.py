from typing import Optional, Union
from ..heads import ChangeHead
from ..model import ChangeModel
from ..encoder import get_encoder
from .FPN import FPNDecoder as FPNDecoder_v1
from .FPN_FPN import FPNDecoder as FPNDecoder_FPN

decoders = {}

decoders.update({"FPN_v1": FPNDecoder_v1})
decoders.update({"FPN_FPN": FPNDecoder_FPN})


class BFPNModel(ChangeModel):

    def __init__(
            self,
            encoder_name: str = "resnet34",
            decoder_name: str = "DSFPN",
            encoder_weights: Optional[str] = "imagenet",
            encode_depth: Optional[int] = 5,
            pyramid_channel: int = 128,
            change_channels: int = 64,
            encoder_feature_merge_policy: str = "sub",
            # decoder_merge_policy: str = "cat",
            in_channels: int = 3,
            classes: int = 1,
    ):
        super().__init__()
        self.encoder_feature_merge_policy = encoder_feature_merge_policy
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            weights=encoder_weights,
        )
        if self.encoder_feature_merge_policy in ("sub-add", "cat"):
            encoder_channels = [2 * i for i in self.encoder.out_channels]
        elif self.encoder_feature_merge_policy in ("sub", "add"):
            encoder_channels = self.encoder.out_channels
        else:
            raise ValueError(
                "`encoder_feature_merge_policy` must be one of: ['add', 'sub', 'cat', 'sub-add'], got {}".format(
                    self.encoder_feature_merge_policy)
            )

        self.decoder = decoders[decoder_name](
            encoder_channels=encoder_channels,
            pyramid_channel=pyramid_channel,
            change_channels=change_channels,
        )
        self.change_head = ChangeHead(
            in_channels=self.decoder.out_channels,
            out_channels=classes,
        )

        self.name = "{}-{}({})".format(encoder_name, decoder_name, encoder_feature_merge_policy)
        self.initialize()
