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
        
        # CIF parameters (from C++ implementation)
        self.cif_threshold = 1.0  # Default CIF threshold
        self.tail_alphas = 0.45   # Default tail alpha value
        self.chunk_size = [5, 10, 5]  # Default chunk size [pre, main, suf]
        
        # CIF cache
        self.hidden_cache = []  # List of hidden state vectors
        self.alphas_cache = []  # List of alpha values
        
        # Initialize decoder cache (for streaming)
        self.decoder_cache = self._init_decoder_cache()
        
    def _cif_search(
        self, 
        hidden: np.ndarray, 
        alphas: np.ndarray, 
        is_final: bool = False
    ) -> np.ndarray:
        """CIF (Continuous Integrate-and-Fire) search to generate acoustic_embeds
        
        This implements the CIF algorithm from paraformer-online.cpp:270-345
        
        Args:
            hidden: Encoder hidden states, shape [T, hidden_dim]
            alphas: Alpha values from encoder, shape [T]
            is_final: Whether this is the final chunk
            
        Returns:
            acoustic_embeds: Generated acoustic embeddings, shape [N, hidden_dim]
            where N is the number of fired frames
        """
        if len(hidden) == 0 or len(alphas) == 0:
            return np.zeros((0, hidden.shape[1] if len(hidden) > 0 else 512), dtype=np.float32)
        
        hidden_size = hidden.shape[1]
        
        # Process chunk_size boundaries (zero out alphas at boundaries)
        chunk_size_pre = self.chunk_size[0]
        chunk_size_suf = sum(self.chunk_size[:-1])
        
        alphas = alphas.copy()
        if chunk_size_pre > 0:
            alphas[:chunk_size_pre] = 0.0
        if chunk_size_suf < len(alphas):
            alphas[chunk_size_suf:] = 0.0
        
        # Merge with cache
        if len(self.hidden_cache) > 0:
            hidden = np.concatenate([np.array(self.hidden_cache), hidden], axis=0)
            alphas = np.concatenate([np.array(self.alphas_cache), alphas], axis=0)
            self.hidden_cache = []
            self.alphas_cache = []
        
        # Add tail for final chunk
        if is_final:
            tail_hidden = np.zeros((1, hidden_size), dtype=np.float32)
            hidden = np.concatenate([hidden, tail_hidden], axis=0)
            alphas = np.concatenate([alphas, np.array([self.tail_alphas])], axis=0)
        
        # CIF integration
        integrate = 0.0
        frames = np.zeros(hidden_size, dtype=np.float32)
        list_frame = []
        
        for i in range(len(alphas)):
            alpha = alphas[i]
            if alpha + integrate < self.cif_threshold:
                integrate += alpha
                frames += alpha * hidden[i]
            else:
                # Fire: emit a frame
                frames += (self.cif_threshold - integrate) * hidden[i]
                list_frame.append(frames.copy())
                integrate += alpha
                integrate -= self.cif_threshold
                frames = integrate * hidden[i]
        
        # Update cache
        if integrate > 0.0:
            self.hidden_cache = [frames / integrate]
            self.alphas_cache = [integrate]
        else:
            self.hidden_cache = [frames]
            self.alphas_cache = [integrate]
        
        if len(list_frame) == 0:
            return np.zeros((0, hidden_size), dtype=np.float32)
        
        return np.array(list_frame, dtype=np.float32)
    
    def _init_decoder_cache(self) -> Dict[str, np.ndarray]:
        """Initialize decoder cache for streaming
        
        FSMN cache shape is [1, fsmn_dims, fsmn_lorder] = [1, 512, 10] by default
        
        Returns:
            Dictionary of decoder cache tensors
        """
        cache = {}
        fsmn_layers = 16  # Default FSMN layers
        fsmn_dims = 512  # Default FSMN dimensions
        fsmn_lorder = 10  # Default FSMN order
        
        for i in range(fsmn_layers):
            cache_name = f'in_cache_{i}'
            if cache_name in self.decoder_input_names:
                # Try to get shape from model
                for inp in self.decoder_session.get_inputs():
                    if inp.name == cache_name:
                        shape = inp.shape
                        # Replace dynamic dims with defaults
                        shape = [1 if (isinstance(s, str) or s < 0) else s for s in shape]
                        if len(shape) == 3:
                            if shape[1] < 0:
                                shape[1] = fsmn_dims
                            if shape[2] < 0:
                                shape[2] = fsmn_lorder
                        elif len(shape) == 0 or any(s < 0 for s in shape):
                            # Default shape if not specified
                            shape = [1, fsmn_dims, fsmn_lorder]
                        cache[cache_name] = np.zeros(shape, dtype=np.float32)
                        break
                else:
                    # If not found in inputs but name pattern matches, use default
                    cache[cache_name] = np.zeros((1, fsmn_dims, fsmn_lorder), dtype=np.float32)
        
        return cache
    
    def reset(self):
        """Reset ASR state"""
        self.frontend.reset()
        self.hidden_cache = []
        self.alphas_cache = []
        self.decoder_cache = self._init_decoder_cache()
    
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
    
    def encode(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run encoder and get hidden states and alpha values
        
        Args:
            features: Input features with shape [n_frames, 560]
            
        Returns:
            Tuple of (encoder_hidden, encoder_alphas)
            - encoder_hidden: [T, hidden_dim]
            - encoder_alphas: [T] - alpha values for CIF
        """
        if len(features) == 0:
            return np.zeros((0, 512), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        
        # Prepare input
        features_input = features.astype(np.float32)[np.newaxis, :, :]  # [1, T, 560]
        features_length = np.array([features.shape[0]], dtype=np.int32)
        
        inputs = {
            'speech': features_input,
            'speech_lengths': features_length,
        }
        
        # Run encoder
        outputs = self.encoder_session.run(self.encoder_output_names, inputs)
        
        from loguru import logger
        
        # Encoder outputs: [0]=enc, [1]=enc_lens, [2]=alphas
        if len(outputs) < 3:
            logger.error(f"Encoder should output at least 3 tensors, got {len(outputs)}")
            return np.zeros((0, 512), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        
        # Extract encoder hidden states [1, T, hidden_dim] -> [T, hidden_dim]
        encoder_hidden = outputs[0]
        if len(encoder_hidden.shape) == 3:
            encoder_hidden = encoder_hidden[0]  # [T, hidden_dim]
        
        # Extract alpha values [1, T] -> [T]
        encoder_alphas = outputs[2]
        if len(encoder_alphas.shape) == 2:
            encoder_alphas = encoder_alphas[0]  # [T]
        elif len(encoder_alphas.shape) == 1:
            encoder_alphas = encoder_alphas
        
        logger.debug(f"Encoder output shapes: hidden={encoder_hidden.shape}, alphas={encoder_alphas.shape}")
        
        return encoder_hidden, encoder_alphas
    
    def decode(
        self, 
        encoder_hidden: np.ndarray, 
        acoustic_embeds: np.ndarray,
        encoder_lens: np.ndarray
    ) -> List[int]:
        """Run decoder to get token predictions
        
        Args:
            encoder_hidden: Encoder hidden states [1, T, hidden_dim]
            acoustic_embeds: Acoustic embeddings from CIF [1, N, hidden_dim]
            encoder_lens: Encoder lengths [1]
            
        Returns:
            List of predicted token IDs
        """
        if len(acoustic_embeds) == 0:
            return []
        
        from loguru import logger
        
        # Prepare decoder inputs in correct order:
        # 1. enc (encoder hidden states)
        # 2. enc_lens (encoder lengths)
        # 3. acoustic_embeds (from CIF search)
        # 4. acoustic_embeds_len
        # 5. decoder cache (in_cache_0, in_cache_1, ...)
        
        inputs = {}
        
        # Add encoder outputs
        if 'enc' in self.decoder_input_names:
            inputs['enc'] = encoder_hidden
            inputs['enc_len'] = encoder_lens
        elif 'encoder_out' in self.decoder_input_names:
            inputs['encoder_out'] = encoder_hidden
            inputs['encoder_out_lens'] = encoder_lens
        
        # Add acoustic_embeds (from CIF search)
        acoustic_embeds_batch = acoustic_embeds.astype(np.float32)[np.newaxis, :, :]  # [1, N, hidden_dim]
        acoustic_embeds_len = np.array([acoustic_embeds.shape[0]], dtype=np.int32)
        
        if 'acoustic_embeds' in self.decoder_input_names:
            inputs['acoustic_embeds'] = acoustic_embeds_batch
            inputs['acoustic_embeds_len'] = acoustic_embeds_len
        
        # Add decoder cache from persistent state
        for cache_name, cache_value in self.decoder_cache.items():
            if cache_name in self.decoder_input_names:
                inputs[cache_name] = cache_value
        
        # Run decoder
        outputs = self.decoder_session.run(self.decoder_output_names, inputs)
        
        logger.debug(f"Decoder ran, got {len(outputs)} outputs")
        
        # Update decoder cache from outputs (FSMN cache is typically outputs[2:])
        # C++ code: decoder_tensor[2+l] for l=0 to fsmn_layers-1
        # Decoder outputs: [0]=logits/sample_ids, [1]=lengths(?), [2:2+fsmn_layers]=FSMN cache
        fsmn_layers = 16  # Default FSMN layers
        cache_start_idx = 2  # First 2 outputs are usually logits/sample_ids and lengths
        
        # Check output names to find cache outputs
        cache_output_names = [name for name in self.decoder_output_names if 'cache' in name.lower()]
        
        if len(cache_output_names) > 0:
            # Use named cache outputs
            for cache_output_name in cache_output_names:
                output_idx = self.decoder_output_names.index(cache_output_name)
                # Extract index from name (e.g., 'out_cache_0' -> 0)
                if '_' in cache_output_name:
                    try:
                        idx = int(cache_output_name.split('_')[-1])
                        input_cache_name = f'in_cache_{idx}'
                        if input_cache_name in self.decoder_cache:
                            self.decoder_cache[input_cache_name] = outputs[output_idx]
                            logger.debug(f"Updated decoder cache {input_cache_name} from {cache_output_name} (shape: {outputs[output_idx].shape})")
                    except ValueError:
                        pass
        else:
            # Fallback: assume cache outputs are at indices [2:2+fsmn_layers]
            for i in range(cache_start_idx, min(len(outputs), cache_start_idx + fsmn_layers)):
                output_idx = i - cache_start_idx
                input_cache_name = f'in_cache_{output_idx}'
                
                if input_cache_name in self.decoder_cache:
                    self.decoder_cache[input_cache_name] = outputs[i]
                    logger.debug(f"Updated decoder cache {input_cache_name} from output[{i}] (shape: {outputs[i].shape})")
        
        # Extract token predictions (first output is usually logits or sample_ids)
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
            # Fallback: use first output (usually logits)
            predictions = outputs[0]
            if len(predictions.shape) == 3:
                predictions = predictions[0]  # [T', vocab_size]
                token_ids = np.argmax(predictions, axis=-1).tolist()
            elif len(predictions.shape) == 2:
                token_ids = np.argmax(predictions, axis=-1).tolist()
            else:
                token_ids = predictions.flatten().tolist()
        
        return token_ids
    
    def infer(self, features: np.ndarray, is_finished: bool = False) -> str:
        """Run full inference pipeline
        
        Args:
            features: Input features with shape [n_frames, 560]
            is_finished: Whether this is the final chunk
            
        Returns:
            Recognized text
        """
        from loguru import logger
        
        # Encode to get hidden states and alphas
        encoder_hidden, encoder_alphas = self.encode(features)
        logger.debug(f"Online encoder output: hidden shape={encoder_hidden.shape}, alphas shape={encoder_alphas.shape}")
        
        # CIF search to generate acoustic_embeds
        acoustic_embeds = self._cif_search(encoder_hidden, encoder_alphas, is_finished=is_finished)
        logger.debug(f"CIF search generated acoustic_embeds: shape={acoustic_embeds.shape}")
        
        if len(acoustic_embeds) == 0:
            return ""
        
        # Prepare decoder inputs
        encoder_hidden_batch = encoder_hidden.astype(np.float32)[np.newaxis, :, :]  # [1, T, hidden_dim]
        encoder_lens = np.array([encoder_hidden.shape[0]], dtype=np.int32)
        
        # Decode
        token_ids = self.decode(encoder_hidden_batch, acoustic_embeds, encoder_lens)
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

