import math

from yacs.config import CfgNode as CN

_c = CN()

_c.display_interval = 25
_c.saved_path = "checkpoints/new_mel"
_c.max_epoch = 25  # 30
_c.text_embd_dims = 768
_c.max_text_num = 900
_c.hidden_size = 512

_c.dataset = CN()
_c.dataset.data_dir = "data"
_c.dataset.batch_size = 10
_c.dataset.max_text_num = _c.max_text_num
_c.dataset.text_embd_dims = _c.text_embd_dims
_c.dataset.max_sample_num = 50
_c.dataset.max_mel_width = 8
_c.dataset.mel_embd_dim = 128

_c.model = CN()
_c.model.num_labels = 4
_c.model.spk_num = 3
_c.model.text_embd_dims = _c.text_embd_dims
_c.model.hidden_size = _c.hidden_size
_c.model.head_num = 8
_c.model.inner_layer = 2
_c.model.outer_layer = 3
_c.model.dropout = 0.4
_c.model.num_labels = 4
_c.model.max_text_num = _c.max_text_num
_c.model.mel_width = _c.dataset.max_mel_width
_c.model.mel_dim = _c.dataset.mel_embd_dim
_c.model.chunk_num = int(math.sqrt(_c.max_text_num))
_c.model.chunk_size = _c.max_text_num // _c.model.chunk_num
_c.model.gate_bias = 0.5
_c.model.use_gate = True
_c.model.use_spk = True
_c.model.use_mel = True
_c.model.use_shift = True
_c.model.use_pivot = True
_c.model.use_init = True
_c.model.use_final = True
_c.model.core = "Pivot"


_c.metric = CN()
_c.metric.split_thresh = 0.25
_c.metric.inside_thresh = 0.7

_c.optimizer = CN()
_c.optimizer.lr = 1e-4
_c.optimizer.weight_decay = 5e-5
_c.optimizer.T_max = _c.max_epoch
_c.optimizer.warmup_epoch = 1

_c.loss = CN()
_c.loss.alpha = 0.75
_c.loss.gamma = 2
_c.loss.chunk_num = _c.model.chunk_num
_c.loss.max_text_num = _c.max_text_num
_c.loss.weight = CN()
_c.loss.weight.split = 1
# _c.loss.weight.conf = 0.4
_c.loss.weight.inside = 0.5
# _c.loss.weight.cls = 1
# _c.loss.weight.score = 0.02




def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _c.clone()
