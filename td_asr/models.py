"""Model loading and inference using ONNX Runtime"""
import onnxruntime as ort
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import yaml
import json
from loguru import logger

class ModelLoader:
    """Base class for model loading"""
    
    def __init__(self, model_dir: Path, quantize: bool = True, thread_num: int = 2):
        self.model_dir = Path(model_dir)
        self.quantize = quantize
        self.thread_num = thread_num
        self.config = None
        self._load_config()
    
    def _load_config(self):
        """Load config.yaml"""
        config_path = self.model_dir / "config.yaml"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        else:
            logger.warning(f"Config file not found: {config_path}")
            self.config = {}
    
    def _load_onnx_model(self, model_name: str) -> ort.InferenceSession:
        """Load ONNX model"""
        if self.quantize:
            model_path = self.model_dir / f"{model_name}_quant.onnx"
            if not model_path.exists():
                model_path = self.model_dir / f"{model_name}.onnx"
        else:
            model_path = self.model_dir / f"{model_name}.onnx"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = self.thread_num
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        providers = ['CPUExecutionProvider']
        
        session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=providers,
        )
        
        logger.info(f"Loaded model: {model_path}")
        return session
    
    def _load_cmvn(self) -> Optional[np.ndarray]:
        """Load CMVN statistics from Kaldi Nnet format"""
        cmvn_path = self.model_dir / "am.mvn"
        if not cmvn_path.exists():
            return None
        
        means = []
        vars = []
        
        try:
            with open(cmvn_path, 'r') as f:
                lines = f.readlines()
                i = 0
                while i < len(lines):
                    line = lines[i].strip()
                    parts = line.split()
                    
                    if parts and parts[0] == "<AddShift>":
                        # Next line should be <LearnRateCoef>
                        i += 1
                        if i < len(lines):
                            means_line = lines[i].strip()
                            means_parts = means_line.split()
                            if means_parts and means_parts[0] == "<LearnRateCoef>":
                                # Extract mean values from index 3 to second-to-last
                                for j in range(3, len(means_parts) - 1):
                                    means.append(float(means_parts[j]))
                    
                    elif parts and parts[0] == "<Rescale>":
                        # Next line should be <LearnRateCoef>
                        i += 1
                        if i < len(lines):
                            vars_line = lines[i].strip()
                            vars_parts = vars_line.split()
                            if vars_parts and vars_parts[0] == "<LearnRateCoef>":
                                # Extract std values from index 3 to second-to-last
                                for j in range(3, len(vars_parts) - 1):
                                    vars.append(float(vars_parts[j]))
                    
                    i += 1
        except Exception as e:
            logger.warning(f"Error loading CMVN from {cmvn_path}: {e}")
            return None
        
        if means and vars:
            # Convert to numpy arrays
            mean_array = np.array(means, dtype=np.float32)
            var_array = np.array(vars, dtype=np.float32)
            # Return as [mean, var] shape (2, feature_dim)
            return np.stack([mean_array, var_array])
        
        return None

class VADModel(ModelLoader):
    """VAD (Voice Activity Detection) model"""
    
    def __init__(self, model_dir: Path, quantize: bool = True, thread_num: int = 2):
        super().__init__(model_dir, quantize, thread_num)
        self.session = self._load_onnx_model("model")
        self.cmvn = self._load_cmvn()
        
        # Get input/output names
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        # VAD state
        self.reset()
    
    def reset(self):
        """Reset VAD state"""
        # Initialize state based on model architecture
        # FSMN-VAD typically uses hidden states
        self.hidden_states = None
    
    def infer(
        self,
        features: np.ndarray,
        is_final: bool = False,
    ) -> Tuple[np.ndarray, bool, bool]:
        """
        Run VAD inference
        
        Args:
            features: Fbank features (shape: [T, n_mels])
            is_final: Whether this is the final chunk
            
        Returns:
            (probs, is_speech, is_endpoint)
            - probs: Speech probabilities
            - is_speech: Whether current frame is speech
            - is_endpoint: Whether endpoint detected
        """
        # Prepare inputs
        inputs = {}
        
        # Features input
        if len(features.shape) == 2:
            features = features[np.newaxis, :, :]  # Add batch dimension
        
        inputs[self.input_names[0]] = features.astype(np.float32)
        
        # Hidden states (if needed)
        if self.hidden_states is not None and len(self.input_names) > 1:
            inputs[self.input_names[1]] = self.hidden_states
        
        # Run inference
        outputs = self.session.run(self.output_names, inputs)
        
        # Update hidden states
        if len(outputs) > 1:
            self.hidden_states = outputs[1]
        
        # Extract probabilities
        probs = outputs[0]  # Shape: [B, T, 2] or [B, T]
        
        if len(probs.shape) == 3:
            # Binary classification: [silence, speech]
            speech_probs = probs[0, :, 1] if probs.shape[2] > 1 else probs[0, :, 0]
        else:
            speech_probs = probs[0, :]
        
        # Threshold for speech detection
        is_speech = speech_probs > 0.5
        
        # Endpoint detection (simplified)
        is_endpoint = False
        if is_final:
            is_endpoint = True
        elif len(is_speech) > 0:
            # Check for silence after speech
            if np.any(is_speech):
                # Simple endpoint: silence for a period
                silence_frames = 0
                for i in range(len(is_speech) - 1, -1, -1):
                    if not is_speech[i]:
                        silence_frames += 1
                    else:
                        break
                # Endpoint if silence > threshold (e.g., 80 frames = 800ms at 10ms/frame)
                is_endpoint = silence_frames > 80
        
        return speech_probs, is_speech, is_endpoint

class OnlineASRModel(ModelLoader):
    """Online ASR model (Paraformer streaming)"""
    
    def __init__(self, model_dir: Path, quantize: bool = True, thread_num: int = 2):
        super().__init__(model_dir, quantize, thread_num)
        
        # Online model: model_quant.onnx is encoder, decoder_quant.onnx is decoder
        self.encoder_session = self._load_onnx_model("model")  # encoder
        self.decoder_session = self._load_onnx_model("decoder")
        
        # Load CMVN
        self.cmvn = self._load_cmvn()
        
        # Load tokens
        self.tokens = self._load_tokens()
        
        # State
        self.reset()
    
    def _load_tokens(self) -> Dict[str, int]:
        """Load token vocabulary"""
        tokens_path = self.model_dir / "tokens.json"
        if tokens_path.exists():
            with open(tokens_path, 'r', encoding='utf-8') as f:
                tokens_data = json.load(f)
                if isinstance(tokens_data, dict) and 'token_list' in tokens_data:
                    token_list = tokens_data['token_list']
                elif isinstance(tokens_data, list):
                    token_list = tokens_data
                else:
                    token_list = []
                
                return {token: idx for idx, token in enumerate(token_list)}
        return {}
    
    def reset(self):
        """Reset model state"""
        # Encoder states (cache)
        self.encoder_cache = None
        # Decoder states
        self.decoder_cache = None
    
    def infer(
        self,
        features: np.ndarray,
        is_final: bool = False,
    ) -> str:
        """
        Run online ASR inference
        
        Args:
            features: Fbank features (shape: [T, n_mels])
            is_final: Whether this is the final chunk
            
        Returns:
            Decoded text
        """
        # Encoder
        encoder_inputs = {}
        encoder_inputs[self.encoder_session.get_inputs()[0].name] = features[np.newaxis, :, :].astype(np.float32)
        
        if self.encoder_cache is not None:
            # Add cache if model expects it
            if len(self.encoder_session.get_inputs()) > 1:
                encoder_inputs[self.encoder_session.get_inputs()[1].name] = self.encoder_cache
        
        encoder_outputs = self.encoder_session.run(None, encoder_inputs)
        
        # Update cache
        if len(encoder_outputs) > 1:
            self.encoder_cache = encoder_outputs[1]
        
        encoder_out = encoder_outputs[0]
        
        # Decoder
        decoder_inputs = {}
        decoder_inputs[self.decoder_session.get_inputs()[0].name] = encoder_out
        
        if self.decoder_cache is not None:
            if len(self.decoder_session.get_inputs()) > 1:
                decoder_inputs[self.decoder_session.get_inputs()[1].name] = self.decoder_cache
        
        decoder_outputs = self.decoder_session.run(None, decoder_inputs)
        
        # Update decoder cache
        if len(decoder_outputs) > 1:
            self.decoder_cache = decoder_outputs[1]
        
        # Decode tokens to text
        logits = decoder_outputs[0]  # Shape: [B, T, vocab_size]
        
        # Greedy decoding
        token_ids = np.argmax(logits[0], axis=-1)
        
        # Convert to text
        text = self._decode_tokens(token_ids)
        
        return text
    
    def _decode_tokens(self, token_ids: np.ndarray) -> str:
        """Decode token IDs to text"""
        # Reverse token dict
        id_to_token = {v: k for k, v in self.tokens.items()}
        
        tokens = []
        for tid in token_ids:
            if tid in id_to_token:
                token = id_to_token[tid]
                # Skip special tokens
                if token not in ['<blank>', '<sos>', '<eos>', '<unk>']:
                    tokens.append(token)
        
        text = ''.join(tokens)
        return text

class OfflineASRModel(ModelLoader):
    """Offline ASR model (Paraformer)"""
    
    def __init__(self, model_dir: Path, quantize: bool = True, thread_num: int = 2):
        super().__init__(model_dir, quantize, thread_num)
        self.session = self._load_onnx_model("model")
        self.cmvn = self._load_cmvn()
        
        # Load tokens
        self.tokens = self._load_tokens()
    
    def _load_tokens(self) -> Dict[str, int]:
        """Load token vocabulary"""
        tokens_path = self.model_dir / "tokens.json"
        if tokens_path.exists():
            with open(tokens_path, 'r', encoding='utf-8') as f:
                tokens_data = json.load(f)
                if isinstance(tokens_data, dict) and 'token_list' in tokens_data:
                    token_list = tokens_data['token_list']
                elif isinstance(tokens_data, list):
                    token_list = tokens_data
                else:
                    token_list = []
                
                return {token: idx for idx, token in enumerate(token_list)}
        return {}
    
    def infer(self, features: np.ndarray) -> str:
        """
        Run offline ASR inference
        
        Args:
            features: Fbank features (shape: [T, n_mels])
            
        Returns:
            Decoded text
        """
        # Prepare input
        if len(features.shape) == 2:
            features = features[np.newaxis, :, :]
        
        inputs = {self.session.get_inputs()[0].name: features.astype(np.float32)}
        
        # Run inference
        outputs = self.session.run(None, inputs)
        
        # Decode
        logits = outputs[0]  # Shape: [B, T, vocab_size]
        token_ids = np.argmax(logits[0], axis=-1)
        
        text = self._decode_tokens(token_ids)
        return text
    
    def _decode_tokens(self, token_ids: np.ndarray) -> str:
        """Decode token IDs to text"""
        id_to_token = {v: k for k, v in self.tokens.items()}
        
        tokens = []
        for tid in token_ids:
            if tid in id_to_token:
                token = id_to_token[tid]
                if token not in ['<blank>', '<sos>', '<eos>', '<unk>']:
                    tokens.append(token)
        
        text = ''.join(tokens)
        return text

class PUNCModel(ModelLoader):
    """Punctuation model (CT-Transformer)"""
    
    def __init__(self, model_dir: Path, quantize: bool = True, thread_num: int = 2):
        super().__init__(model_dir, quantize, thread_num)
        self.session = self._load_onnx_model("model")
        self.tokens = self._load_tokens()
        self.punc_tokens = ['，', '。', '？', '！', '、']  # Common punctuation
    
    def _load_tokens(self) -> Dict[str, int]:
        """Load token vocabulary"""
        tokens_path = self.model_dir / "tokens.json"
        if tokens_path.exists():
            with open(tokens_path, 'r', encoding='utf-8') as f:
                tokens_data = json.load(f)
                if isinstance(tokens_data, dict) and 'token_list' in tokens_data:
                    return {token: idx for idx, token in enumerate(tokens_data['token_list'])}
                elif isinstance(tokens_data, list):
                    return {token: idx for idx, token in enumerate(tokens_data)}
        return {}
    
    def infer(self, text: str) -> str:
        """
        Add punctuation to text
        
        Args:
            text: Input text without punctuation
            
        Returns:
            Text with punctuation
        """
        if not text:
            return text
        
        # Convert text to token IDs
        token_ids = []
        for char in text:
            if char in self.tokens:
                token_ids.append(self.tokens[char])
            else:
                token_ids.append(self.tokens.get('<unk>', 0))
        
        if not token_ids:
            return text
        
        # Prepare input
        inputs = {self.session.get_inputs()[0].name: np.array([token_ids], dtype=np.int64)}
        
        # Run inference
        outputs = self.session.run(None, inputs)
        
        # Decode with punctuation
        logits = outputs[0]  # Shape: [B, T, vocab_size]
        pred_ids = np.argmax(logits[0], axis=-1)
        
        # Convert back to text with punctuation
        id_to_token = {v: k for k, v in self.tokens.items()}
        result = []
        for i, (char, pid) in enumerate(zip(text, pred_ids)):
            result.append(char)
            if pid in id_to_token:
                punc = id_to_token[pid]
                if punc in self.punc_tokens:
                    result.append(punc)
        
        return ''.join(result)

