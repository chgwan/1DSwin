from .SwinCNN_1d import Swin1dClass
from .SwinCNN_1d import Swin1dSeq, Swin1dSeqMultipleLayer

def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == "swin1d_class":
        model = Swin1dClass(
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            input_dim=config.MODEL.SWIN1d_CLASS.IN_CHANS,
            embed_dim=config.MODEL.SWIN1d_CLASS.EMBED_DIM,
            depths=config.MODEL.SWIN1d_CLASS.DEPTHS,
            num_heads=config.MODEL.SWIN1d_CLASS.NUM_HEADS,
            window_size=config.MODEL.SWIN1d_CLASS.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN1d_CLASS.MLP_RATIO,
            dropout=config.MODEL.SWIN1d_CLASS.DROPOUT,
            attention_dropout=config.MODEL.SWIN1d_CLASS.ATTENTION_DROPOUT,
            stochastic_depth_prob=config.MODEL.SWIN1d_CLASS.STOCHASTIC_DEPTH_PROB,
            num_classes=config.MODEL.SWIN1d_CLASS.NUM_CLASS,
            use_checkpoint=config.MODEL.SWIN1d_CLASS.USE_CHECKPOINT,
        )
    elif model_type == "swin1d_seq":
        model = Swin1dSeq(
            input_dim=config.MODEL.SWIN1d_SEQ.IN_CHANS,
            embed_dim=config.MODEL.EMBED_DIM,
            depths=config.MODEL.SWIN1d_SEQ.DEPTHS,
            num_heads=config.MODEL.SWIN1d_SEQ.NUM_HEADS,
            window_size=config.MODEL.SWIN1d_SEQ.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN1d_SEQ.MLP_RATIO,
            dropout=config.MODEL.SWIN1d_SEQ.DROPOUT,
            attention_dropout=config.MODEL.SWIN1d_SEQ.ATTENTION_DROPOUT,
            stochastic_depth_prob=config.MODEL.SWIN1d_SEQ.STOCHASTIC_DEPTH_PROB,
            use_checkpoint=config.MODEL.SWIN1d_SEQ.USE_CHECKPOINT,
        )        
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")
    return model

def build_model_from_dict(config):
    model_type = config['model_type']
    if model_type == "swin1d_class":
        model = Swin1dClass(
            patch_size=config['patch_size'],
            input_dim=config['input_dim'],
            embed_dim=config['embed_dim'],
            depths=config['depths'],
            num_heads=config['num_heads'],
            window_size=config['window_size'],
            mlp_ratio=config.get("mlp_ratio", 4.0),
            dropout=config.get("dropout", 0.0),
            attention_dropout=config.get("attention_dropout", 0.0),
            stochastic_depth_prob=config.get("stochastic_depth_prob", 0.0),
            num_classes=config.get("num_classes", 1000),
            norm_layer=config.get("norm_layer", None),
            use_checkpoint=config.get("use_checkpoint", False),
        )
    elif model_type == "swin1d_seq":
        model = Swin1dSeqMultipleLayer(
            input_dim=config['input_dim'],
            embed_dim=config['embed_dim'],
            output_dim=config['output_dim'],
            depths=config['depths'],
            num_heads=config['num_heads'],
            window_size=config['window_size'],
            num_layers=config.get('num_layers', 1),
            mlp_ratio=config.get("mlp_ratio", 4.0),
            dropout=config.get("dropout", 0.0),
            attention_dropout=config.get("attention_dropout", 0.0),
            stochastic_depth_prob=config.get("stochastic_depth_prob", 0.0),
            norm_layer=config.get("norm_layer", None),
            block=config.get("block", None),
            use_checkpoint=config.get("use_checkpoint", False),
        )  
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")
    return model