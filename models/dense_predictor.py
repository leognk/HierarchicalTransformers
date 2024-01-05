from .type import ModelType
from .encodecos import create_encodeco
from .ssl import create_ssl
from .heads.dense_predictor import create_head
from .utils import BackboneWithHead
import utils


class DensePredictor(BackboneWithHead):

    def __init__(self, backbone, head):
        super().__init__(backbone, head)
        self.type = ModelType.DENSE_PREDICTOR
    
    def forward(self, x):
        x, cp = self.backbone(x)
        x = self.head(x, cp)
        return x


def create_dense_predictor(
    encodeco_from_ssl,
    encodeco_config_dir,
    encodeco_config,
    head_config_dir,
    head_config,
    in_channels,
    in_size,
    out_channels,
    init_state_dict=None,
    init_head=None,
):
    """
    dense_predictor:
        - I/O: img -> pred
        - dense_predictor = encodeco + head
        - shape: [b c N1 ... Nd] -> [b c' N1 ... Nd]
            where c=in_channels, N=in_size, c'=out_channels
    Args:
        - init_state_dict (dict, optional): model state dict to initialize from
        - init_head (bool, optional): whether to initialize the head from the state dict
    """
    if init_state_dict:
        init_type = ModelType(init_state_dict['type'])
        init_params = init_state_dict['params']
    else:
        init_type, init_params = None, None

    if not encodeco_from_ssl:
        encodeco = create_encodeco(encodeco_config_dir, encodeco_config, in_channels, in_size)
    else:
        ssl = create_ssl(encodeco_config_dir, encodeco_config, in_channels, in_size)
        if init_type == ModelType.SSL:
            ssl.load_state_dict(init_params)
        encodeco = ssl.create_encodeco()
        encodeco.args = ssl.args
    
    head = create_head(
        head_config_dir,
        head_config,
        in_dim=encodeco.out_dim,
        out_dim=out_channels,
        patch_size=encodeco.out_patch_size,
        pre_norm=encodeco.head_pre_norm,
    )

    if init_type == ModelType.DENSE_PREDICTOR:
        encodeco_sd = utils.filter_state_dict(init_params, 'backbone')
        encodeco.load_state_dict(encodeco_sd)
        if init_head:
            head_sd = utils.filter_state_dict(init_params, 'head')
            head.load_state_dict(head_sd)

    return DensePredictor(encodeco, head)