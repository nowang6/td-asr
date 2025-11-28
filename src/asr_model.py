"""ASR Models for Online and Offline Recognition"""

import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from .utils import load_yaml_config, load_cmvn, apply_cmvn, load_tokens
from .frontend import WavFrontendOnline


class OnlineASRModel:
    """Paraformer Online ASR Model
    
    This model uses a streaming encoder-decoder architecture with caching
    for real-time speech recognition.
    
    Args:
        model_dir: Path to online ASR model directory
        quantize: Whether to use quantized model (default: True)
    """
    
    def __init__(self, model_dir: str, quantize: bool = True):
        self.model_dir = Path(model_dir)
        self.quantize = quantize
        
        # Load config
        config_path = self.model_dir / "config.yaml"
        self.config = load_yaml_config(config_path)
        
        # Load CMVN
        cmvn_path = self.model_dir / "am.mvn"
        self.cmvn = load_cmvn(cmvn_path)
        
        # Load tokens
        token_path = self.model_dir / "tokens.json"
        if not token_path.exists():
            # Try tokens.txt
            token_path = self.model_dir / "tokens.txt"
        
        self.tokens = []
        if token_path.exists():
            if str(token_path).endswith('.json'):
                import json
                with open(token_path, 'r', encoding='utf-8') as f:
                    token_data = json.load(f)
                    # tokens.json is a simple array
                    if isinstance(token_data, list):
                        self.tokens = token_data
                    elif isinstance(token_data, dict):
                        self.tokens = token_data.get('token_list', [])
            else:
                self.tokens = load_tokens(token_path)
        
        # Create frontend for ASR (lfr_m=7, lfr_n=6 for 560-dim features)
        frontend_conf = self.config.get('frontend_conf', {})
        self.frontend = WavFrontendOnline(
            fs=frontend_conf.get('fs', 16000),
            window=frontend_conf.get('window', 'hamming'),
            n_mels=frontend_conf.get('n_mels', 80),
            frame_length=frontend_conf.get('frame_length', 25),
            frame_shift=frontend_conf.get('frame_shift', 10),
            dither=frontend_conf.get('dither', 0.0),
            lfr_m=frontend_conf.get('lfr_m', 7),
            lfr_n=frontend_conf.get('lfr_n', 6),
        )
        
        # Load encoder ONNX model
        encoder_file = "model_quant.onnx" if quantize else "model.onnx"
        encoder_path = self.model_dir / encoder_file
        
        sess_options = ort.SessionOptions()
        sess_options.inter_op_num_threads = 2
        sess_options.intra_op_num_threads = 2
        
        self.encoder_session = ort.InferenceSession(
            str(encoder_path),
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        
        # Load decoder ONNX model
        decoder_file = "decoder_quant.onnx" if quantize else "decoder.onnx"
        decoder_path = self.model_dir / decoder_file
        
        self.decoder_session = ort.InferenceSession(
            str(decoder_path),
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        
        # Get model input/output names
        self.encoder_input_names = [inp.name for inp in self.encoder_session.get_inputs()]
        self.encoder_output_names = [out.name for out in self.encoder_session.get_outputs()]
        self.decoder_input_names = [inp.name for inp in self.decoder_session.get_inputs()]
        self.decoder_output_names = [out.name for out in self.decoder_session.get_outputs()]
        
        # Log model I/O for debugging
        from loguru import logger
        logger.debug(f"Online ASR Encoder inputs: {self.encoder_input_names}")
        logger.debug(f"Online ASR Encoder outputs: {self.encoder_output_names}")
        logger.debug(f"Online ASR Decoder inputs: {self.decoder_input_names}")
        logger.debug(f"Online ASR Decoder outputs: {self.decoder_output_names}")
        
        # Initialize encoder cache
        self.encoder_cache = self._init_encoder_cache()
        
    def _init_encoder_cache(self) -> Dict[str, np.ndarray]:
        """Initialize encoder cache for streaming
        
        The online model uses cache to maintain state across chunks.
        Default cache shape is [1, 512, 10] (batch, hidden_dim, cache_len)
        
        Returns:
            Dictionary of cache tensors
        """
        cache = {}
        
        # Check if cache inputs are defined in model
        cache_names = ['in_cache0', 'in_cache1', 'in_cache2', 'in_cache3']
        
        for name in cache_names:
            if name in self.encoder_input_names:
                # Get shape from model input
                for inp in self.encoder_session.get_inputs():
                    if inp.name == name:
                        shape = inp.shape
                        # Replace dynamic dims with default values
                        shape = [1 if (isinstance(s, str) or s < 0) else s for s in shape]
                        if len(shape) == 3 and shape[1] < 0:
                            shape[1] = 512  # Default hidden_dim
                        if len(shape) == 3 and shape[2] < 0:
                            shape[2] = 10  # Default cache_len
                        cache[name] = np.zeros(shape, dtype=np.float32)
                        break
            else:
                # Create default cache if not in model inputs
                cache[name] = np.zeros((1, 512, 10), dtype=np.float32)
        
        return cache
    
    def reset(self):
        """Reset ASR state"""
        self.frontend.reset()
        self.encoder_cache = self._init_encoder_cache()
    
    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract features for ASR
        
        Args:
            audio: Audio waveform
            
        Returns:
            Features with shape [n_frames, 560]
        """
        # Extract fbank features
        features = self.frontend.extract_feat(audio)
        
        # Apply CMVN
        if len(features) > 0:
            features = apply_cmvn(features, self.cmvn)
        
        return features
    
    def encode(self, features: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Run encoder with caching
        
        Args:
            features: Input features with shape [n_frames, 560]
            
        Returns:
            Tuple of (encoder_output, updated_cache)
        """
        if len(features) == 0:
            return np.zeros((0, 512), dtype=np.float32), self.encoder_cache
        
        # Prepare input
        features_input = features.astype(np.float32)[np.newaxis, :, :]  # [1, T, 560]
        features_length = np.array([features.shape[0]], dtype=np.int32)
        
        # Prepare cache inputs
        inputs = {
            'speech': features_input,
            'speech_lengths': features_length,
        }
        
        # Add cache inputs
        for cache_name, cache_value in self.encoder_cache.items():
            if cache_name in self.encoder_input_names:
                inputs[cache_name] = cache_value
        
        # Run encoder
        outputs = self.encoder_session.run(self.encoder_output_names, inputs)
        
        # Extract encoder output (first output is usually the hidden states)
        encoder_output = outputs[0]
        if len(encoder_output.shape) == 3:
            encoder_output = encoder_output[0]  # [T', hidden_dim]
        
        # Update cache from outputs
        updated_cache = {}
        for i, output_name in enumerate(self.encoder_output_names):
            if 'cache' in output_name.lower():
                # Map output cache to input cache
                # e.g., 'out_cache0' -> 'in_cache0' or 'cache_0' -> 'in_cache_0'
                if output_name.startswith('out_'):
                    input_name = output_name.replace('out_', 'in_')
                else:
                    input_name = 'in_' + output_name
                updated_cache[input_name] = outputs[i]
        
        # If no cache outputs found, keep old cache
        if not updated_cache:
            updated_cache = self.encoder_cache
        
        return encoder_output, updated_cache
    
    def decode(self, encoder_output: np.ndarray) -> List[int]:
        """Run decoder to get token predictions
        
        Args:
            encoder_output: Encoder output with shape [T, hidden_dim]
            
        Returns:
            List of predicted token IDs
        """
        if len(encoder_output) == 0:
            return []
        
        # Prepare input
        encoder_out = encoder_output.astype(np.float32)[np.newaxis, :, :]  # [1, T, hidden_dim]
        encoder_out_lens = np.array([encoder_output.shape[0]], dtype=np.int32)
        
        # Check decoder input names to use correct naming
        inputs = {}
        
        # The decoder model expects different input names
        if 'enc' in self.decoder_input_names:
            inputs['enc'] = encoder_out
            inputs['enc_len'] = encoder_out_lens
        elif 'encoder_out' in self.decoder_input_names:
            inputs['encoder_out'] = encoder_out
            inputs['encoder_out_lens'] = encoder_out_lens
        
        # Add acoustic_embeds if required (usually zeros for non-contextual models)
        if 'acoustic_embeds' in self.decoder_input_names:
            # Create dummy acoustic embeds (usually not used in standard paraformer)
            acoustic_embeds = np.zeros((1, 1, 512), dtype=np.float32)
            inputs['acoustic_embeds'] = acoustic_embeds
            inputs['acoustic_embeds_len'] = np.array([1], dtype=np.int32)
        
        # Add decoder cache if required (usually for streaming decoder)
        # Get cache shape from model input if available
        for i in range(16):
            cache_name = f'in_cache_{i}'
            if cache_name in self.decoder_input_names:
                # Try to get shape from model
                cache_shape = None
                for inp in self.decoder_session.get_inputs():
                    if inp.name == cache_name:
                        cache_shape = inp.shape
                        break
                
                if cache_shape:
                    # Replace dynamic dims with defaults
                    cache_shape = [1 if (isinstance(s, str) or s < 0) else s for s in cache_shape]
                    inputs[cache_name] = np.zeros(cache_shape, dtype=np.float32)
                else:
                    # Default cache shape
                    inputs[cache_name] = np.zeros((1, 512, 10), dtype=np.float32)
        
        # Run decoder
        outputs = self.decoder_session.run(self.decoder_output_names, inputs)
        
        # Check if 'sample_ids' is in outputs (decoder may directly output token IDs)
        sample_ids_output = None
        logits_output = None
        
        for i, name in enumerate(self.decoder_output_names):
            if name == 'sample_ids':
                sample_ids_output = outputs[i]
            elif name == 'logits':
                logits_output = outputs[i]
        
        # Prefer sample_ids if available (already token IDs)
        if sample_ids_output is not None:
            token_ids = sample_ids_output.flatten().tolist()
        elif logits_output is not None:
            # Use logits with argmax
            predictions = logits_output
            if len(predictions.shape) == 3:
                predictions = predictions[0]  # [T', vocab_size]
            token_ids = np.argmax(predictions, axis=-1).tolist()
        else:
            # Fallback: use first output
            predictions = outputs[0]
            if len(predictions.shape) == 3:
                predictions = predictions[0]
                token_ids = np.argmax(predictions, axis=-1).tolist()
            elif len(predictions.shape) == 2:
                token_ids = np.argmax(predictions, axis=-1).tolist()
            else:
                token_ids = predictions.flatten().tolist()
        
        return token_ids
    
    def infer(self, features: np.ndarray) -> str:
        """Run full inference pipeline
        
        Args:
            features: Input features with shape [n_frames, 560]
            
        Returns:
            Recognized text
        """
        from loguru import logger
        
        # Encode
        encoder_output, self.encoder_cache = self.encode(features)
        logger.debug(f"Online encoder output: shape={encoder_output.shape}")
        
        # Decode
        token_ids = self.decode(encoder_output)
        logger.debug(f"Online decoder tokens: {len(token_ids)} tokens, first 20: {token_ids[:20]}")
        
        # Convert to text
        text = self.tokens_to_text(token_ids)
        logger.debug(f"Online ASR text: '{text}'")
        
        return text
    
    def tokens_to_text(self, token_ids: List[int]) -> str:
        """Convert token IDs to text
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Text string
        """
        if not self.tokens:
            return ''.join([str(tid) for tid in token_ids])
        
        from loguru import logger
        
        text_parts = []
        for tid in token_ids:
            if 0 <= tid < len(self.tokens):
                token = self.tokens[tid]
                # Skip special tokens (check the actual token strings)
                if token in ['<blank>', '<unk>', '<s>', '</s>', '<pad>', '<eos>', '<sos>']:
                    continue
                # Skip if token starts with < and ends with > (special tokens)
                if token.startswith('<') and token.endswith('>'):
                    continue
                text_parts.append(token)
            else:
                logger.warning(f"Token ID {tid} out of range (vocab size: {len(self.tokens)})")
        
        result = ''.join(text_parts)
        logger.debug(f"Converted {len(token_ids)} tokens to text: '{result}'")
        return result


class OfflineASRModel:
    """Paraformer Offline ASR Model
    
    This model processes complete utterances for higher accuracy.
    
    Args:
        model_dir: Path to offline ASR model directory
        quantize: Whether to use quantized model (default: True)
    """
    
    def __init__(self, model_dir: str, quantize: bool = True):
        self.model_dir = Path(model_dir)
        self.quantize = quantize
        
        # Load config
        config_path = self.model_dir / "config.yaml"
        self.config = load_yaml_config(config_path)
        
        # Load CMVN
        cmvn_path = self.model_dir / "am.mvn"
        self.cmvn = load_cmvn(cmvn_path)
        
        # Load tokens
        token_path = self.model_dir / "tokens.json"
        if not token_path.exists():
            token_path = self.model_dir / "tokens.txt"
        
        self.tokens = []
        if token_path.exists():
            if str(token_path).endswith('.json'):
                import json
                with open(token_path, 'r', encoding='utf-8') as f:
                    token_data = json.load(f)
                    # tokens.json is a simple array
                    if isinstance(token_data, list):
                        self.tokens = token_data
                    elif isinstance(token_data, dict):
                        self.tokens = token_data.get('token_list', [])
            else:
                self.tokens = load_tokens(token_path)
        
        # Create frontend for offline ASR
        # Offline models typically use similar parameters to online
        frontend_conf = self.config.get('frontend_conf', {})
        self.frontend = WavFrontendOnline(
            fs=frontend_conf.get('fs', 16000),
            window=frontend_conf.get('window', 'hamming'),
            n_mels=frontend_conf.get('n_mels', 80),
            frame_length=frontend_conf.get('frame_length', 25),
            frame_shift=frontend_conf.get('frame_shift', 10),
            dither=frontend_conf.get('dither', 0.0),
            lfr_m=frontend_conf.get('lfr_m', 7),
            lfr_n=frontend_conf.get('lfr_n', 6),
        )
        
        # Load ONNX model
        model_file = "model_quant.onnx" if quantize else "model.onnx"
        model_path = self.model_dir / model_file
        
        sess_options = ort.SessionOptions()
        sess_options.inter_op_num_threads = 2
        sess_options.intra_op_num_threads = 2
        
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        
        # Get model input/output names
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        # Log model I/O for debugging
        from loguru import logger
        logger.debug(f"Offline ASR inputs: {self.input_names}")
        logger.debug(f"Offline ASR outputs: {self.output_names}")
    
    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract features for ASR
        
        Args:
            audio: Audio waveform
            
        Returns:
            Features with shape [n_frames, feature_dim]
        """
        # Extract fbank features
        features = self.frontend.extract_feat(audio)
        
        # Apply CMVN
        if len(features) > 0:
            features = apply_cmvn(features, self.cmvn)
        
        return features
    
    def infer(self, features: np.ndarray) -> str:
        """Run inference on complete utterance
        
        Args:
            features: Input features
            
        Returns:
            Recognized text
        """
        if len(features) == 0:
            return ""
        
        # Prepare input
        features_input = features.astype(np.float32)[np.newaxis, :, :]  # [1, T, D]
        features_length = np.array([features.shape[0]], dtype=np.int32)
        
        inputs = {
            'speech': features_input,
            'speech_lengths': features_length,
        }
        
        # Run inference
        outputs = self.session.run(self.output_names, inputs)
        
        # Extract predictions
        # Output format depends on model, typically [1, T', vocab_size]
        predictions = outputs[0][0]  # [T', vocab_size] or [T']
        
        if len(predictions.shape) == 2:
            # [T', vocab_size] - need argmax
            token_ids = np.argmax(predictions, axis=-1).tolist()
        else:
            # [T'] - already token IDs
            token_ids = predictions.tolist()
        
        # Convert to text
        text = self.tokens_to_text(token_ids)
        
        return text
    
    def tokens_to_text(self, token_ids: List[int]) -> str:
        """Convert token IDs to text
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Text string
        """
        if not self.tokens:
            return ''.join([str(tid) for tid in token_ids])
        
        text_parts = []
        for tid in token_ids:
            if isinstance(tid, (list, np.ndarray)):
                tid = int(tid[0]) if len(tid) > 0 else 0
            tid = int(tid)
            
            if 0 <= tid < len(self.tokens):
                token = self.tokens[tid]
                # Skip special tokens
                if token not in ['<blank>', '<unk>', '<s>', '</s>', '<pad>']:
                    text_parts.append(token)
        
        return ''.join(text_parts)

