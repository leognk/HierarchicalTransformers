from .type import ModelType
from .encoders import create_encoder
from .ssl import create_ssl
from .heads.classifier import create_head
from .utils import BackboneWithHead
import utils


class Classifier(BackboneWithHead):

    def __init__(self, backbone, head):
        super().__init__(backbone, head)
        self.type = ModelType.CLASSIFIER
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


def create_classifier(
    enc_from_ssl,
    enc_config_dir,
    enc_config,
    head_config_dir,
    head_config,
    in_channels,
    in_size,
    n_classes,
    init_state_dict=None,
    init_head=None,
):
    """
    classifier:
        - I/O: img -> pred
        - classifier = encoder + head
        - shape: [b c N1 ... Nd] -> [b c']
            where c=in_channels, N=in_size, c'=n_classes
    Args:
        - init_state_dict (dict, optional): model state dict to initialize from
        - init_head (bool, optional): whether to initialize the head from the state dict
    """
    if init_state_dict:
        init_type = ModelType(init_state_dict['type'])
        init_params = init_state_dict['params']
    else:
        init_type, init_params = None, None

    if not enc_from_ssl:
        enc = create_encoder(enc_config_dir, enc_config, in_channels, in_size, n_classes)
    else:
        ssl = create_ssl(enc_config_dir, enc_config, in_channels, in_size)
        if init_type == ModelType.SSL:
            ssl.load_state_dict(init_params)
        enc = ssl.create_encoder()
        enc.args = ssl.args

    head = create_head(head_config_dir, head_config, in_dim=enc.out_dim, out_dim=n_classes, pre_norm=enc.head_pre_norm)

    if init_type == ModelType.CLASSIFIER:
        enc_sd = utils.filter_state_dict(init_params, 'backbone')
        enc.load_state_dict(enc_sd)
        if init_head:
            head_sd = utils.filter_state_dict(init_params, 'head')
            head.load_state_dict(head_sd)
    
    return Classifier(enc, head)