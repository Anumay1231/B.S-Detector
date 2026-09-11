"""SLS (Selective Layer Summarization) classifier for audio deepfake detection.

Architecture (must match checkpoint from sukhdeveyash/XLS-R-SLS-Deepfake-Detection):
  XLS-R 300M (frozen, via HuggingFace transformers) ->
  Per-layer sigmoid attention gating ->
  Weighted layer fusion ->
  BatchNorm2d -> SELU -> MaxPool2d(3,3) ->
  FC(22847->1024) -> SELU -> FC(1024->2) -> LogSoftmax

The pretrained checkpoint stores the full model state_dict with keys like:
  ssl_model.model.* (backbone - we skip these)
  fc0.weight, fc0.bias (attention gate)
  first_bn.weight/bias/running_mean/running_var/num_batches_tracked
  fc1.weight, fc1.bias
  fc3.weight, fc3.bias
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model
from huggingface_hub import hf_hub_download
import logging

logger = logging.getLogger(__name__)

INPUT_LENGTH = 64600  # Fixed input: 64,600 samples at 16kHz (~4.04 seconds)
NUM_LAYERS = 24       # XLS-R 300M has 24 transformer layers
HIDDEN_DIM = 1024     # Hidden dimension of XLS-R


class SLSHead(nn.Module):
    """SLS classifier head.
    
    Takes hidden states from all 24 transformer layers, applies learned
    per-layer sigmoid attention gating, fuses layers, then classifies
    via BatchNorm -> MaxPool -> FC layers.
    """
    
    def __init__(self):
        super().__init__()
        # Per-layer attention gate: pool each layer's output across time,
        # then project to scalar and apply sigmoid
        self.fc0 = nn.Linear(HIDDEN_DIM, 1)  # 1024 -> 1
        
        # Post-fusion processing
        self.first_bn = nn.BatchNorm2d(1)
        self.selu = nn.SELU()
        
        # MaxPool2d(3,3) on (1, 201, 1024) -> (1, 67, 341)
        # Flattened: 67 * 341 = 22847
        self.fc1 = nn.Linear(22847, 1024)
        self.fc3 = nn.Linear(1024, 2)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, hidden_states_list):
        """Forward pass through SLS head.
        
        Args:
            hidden_states_list: List of 24 tensors, each (B, T, 1024)
                representing transformer layer outputs.
        
        Returns:
            log_probs: (B, 2) log-probabilities [spoof, bonafide]
        """
        B = hidden_states_list[0].shape[0]
        
        # Step 1: Compute per-layer attention weights
        weights = []
        for layer_out in hidden_states_list:
            # Average pool across time: (B, T, 1024) -> (B, 1024)
            pooled = layer_out.mean(dim=1)
            # Project to scalar and apply sigmoid: (B, 1024) -> (B, 1)
            w = torch.sigmoid(self.fc0(pooled))
            weights.append(w)
        
        # Step 2: Weighted fusion of all layers
        # Each weight is (B, 1), each layer_out is (B, T, 1024)
        fused = torch.zeros_like(hidden_states_list[0])  # (B, T, 1024)
        for w, layer_out in zip(weights, hidden_states_list):
            # w: (B, 1) -> (B, 1, 1) for broadcasting
            fused = fused + w.unsqueeze(-1) * layer_out
        
        # Step 3: Reshape for 2D processing: (B, 1, T, 1024)
        fused = fused.unsqueeze(1)
        
        # Step 4: BatchNorm -> SELU -> MaxPool
        x = self.first_bn(fused)
        x = self.selu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=3)
        
        # Step 5: Flatten and classify
        x = x.flatten(start_dim=1)  # (B, 22847)
        x = self.selu(self.fc1(x))
        x = self.fc3(x)
        x = self.logsoftmax(x)
        
        return x


def load_sls_checkpoint(sls_head: SLSHead, checkpoint_version='v1'):
    """Load pretrained SLS head weights from HuggingFace Hub.
    
    Downloads the checkpoint from sukhdeveyash/XLS-R-SLS-Deepfake-Detection
    and loads only the SLS head weights (skipping backbone weights).
    
    Args:
        sls_head: SLSHead module to load weights into
        checkpoint_version: 'v1' or 'v2' (v1 recommended)
    """
    if checkpoint_version == 'v1':
        filename = 'v1/epoch_2.pth'
    else:
        filename = 'v2/epoch_16.pth'
    
    logger.info(f'Downloading SLS checkpoint: {filename}')
    ckpt_path = hf_hub_download(
        repo_id='sukhdeveyash/XLS-R-SLS-Deepfake-Detection',
        filename=filename
    )
    
    logger.info(f'Loading checkpoint from {ckpt_path}')
    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    # Extract only SLS head keys (strip 'module.' if present, skip ssl_model.* backbone keys)
    sls_keys = {}
    skipped = 0
    for key, value in state_dict.items():
        clean_key = key
        if clean_key.startswith('module.'):
            clean_key = clean_key[len('module.'):]
        if clean_key.startswith('ssl_model.'):
            skipped += 1
            continue
        sls_keys[clean_key] = value
    
    logger.info(f'Checkpoint has {len(state_dict)} keys total, '
                f'skipped {skipped} backbone keys, '
                f'loading {len(sls_keys)} SLS head keys')
    logger.info(f'SLS head keys: {list(sls_keys.keys())}')
    
    # Load into SLS head
    missing, unexpected = sls_head.load_state_dict(sls_keys, strict=False)
    if missing:
        logger.warning(f'Missing keys in SLS head: {missing}')
    if unexpected:
        logger.warning(f'Unexpected keys: {unexpected}')
    
    logger.info('SLS head weights loaded successfully')
    return sls_head


def load_backbone(device='cpu'):
    """Load frozen XLS-R 300M backbone from HuggingFace.
    
    Returns the model set to eval mode with all parameters frozen.
    """
    logger.info('Loading XLS-R 300M backbone from HuggingFace...')
    backbone = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-xls-r-300m')
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False
    backbone = backbone.to(device)
    logger.info(f'Backbone loaded on {device}, '
                f'{sum(p.numel() for p in backbone.parameters())/1e6:.1f}M params (all frozen)')
    return backbone


def extract_hidden_states(backbone, waveform, device='cpu'):
    """Extract hidden states from all 24 transformer layers.
    
    Args:
        backbone: Wav2Vec2Model
        waveform: (B, T) tensor of raw audio at 16kHz, padded/truncated to INPUT_LENGTH
        device: device to run on
    
    Returns:
        List of 24 tensors, each (B, T_out, 1024) where T_out depends on input length.
        For INPUT_LENGTH=64600, T_out=201.
    """
    waveform = waveform.to(device)
    with torch.no_grad():
        outputs = backbone(waveform, output_hidden_states=True)
    # outputs.hidden_states is a tuple of 25 tensors:
    # index 0 = CNN feature extractor output
    # index 1..24 = transformer layer outputs
    # We want layers 1..24 (the 24 transformer layers)
    hidden_states = list(outputs.hidden_states[1:])  # 24 tensors, each (B, T, 1024)
    return hidden_states


def pad_or_truncate(waveform, target_length=INPUT_LENGTH):
    """Pad (by tiling) or truncate waveform to fixed length.
    
    Args:
        waveform: 1D tensor of audio samples
        target_length: target number of samples
    
    Returns:
        1D tensor of exactly target_length samples
    """
    if waveform.shape[0] >= target_length:
        return waveform[:target_length]
    # Tile-pad: repeat the waveform until we reach target length
    repeats = (target_length // waveform.shape[0]) + 1
    tiled = waveform.repeat(repeats)
    return tiled[:target_length]
