from .SwinCNN_1d import Swin1dClass
from .SwinCNN_1d import Swin1dSeq

def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == "swin1d_class":
        model = Swin1dClass(
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            input_dim=config.MODEL.SWIN1d_CLASS.IN_CHANS,
            embed_dim=config.MODEL.SWIN1d_CLASS.NUM_CLASSES,
            depths=config.MODEL.SWIN1d_CLASS.DEPTHS,
            num_heads=config.MODEL.SWIN1d_CLASS.NUM_HEADS,
            window_size=config.MODEL.SWIN1d_CLASS.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN1d_CLASS.MLP_RATIO,
            dropout=config.MODEL.SWIN1d_CLASS.DROPOUT,
            attention_dropout=config.MODEL.SWIN1d_CLASS.ATTENTION_DROPOUT,
            stochastic_depth_prob=config.MODEL.SWIN1d_CLASS.STOCHASTIC_DEPTH_PROB,
            num_classes=config.MODEL.SWIN1d_CLASS.NUM_CLASS,
        )
    elif model_type == "swin1d_seq":
        model = Swin1dClass(
            input_dim=config.MODEL.SWIN1d_SEQ.IN_CHANS,
            embed_dim=config.MODEL.EMBED_DIM,
            depths=config.MODEL.SWIN1d_SEQ.DEPTHS,
            num_heads=config.MODEL.SWIN1d_SEQ.NUM_HEADS,
            window_size=config.MODEL.SWIN1d_SEQ.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN1d_SEQ.MLP_RATIO,
            dropout=config.MODEL.SWIN1d_SEQ.DROPOUT,
            attention_dropout=config.MODEL.SWIN1d_SEQ.ATTENTION_DROPOUT,
            stochastic_depth_prob=config.MODEL.SWIN1d_SEQ.STOCHASTIC_DEPTH_PROB,
        )        
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")
    return model