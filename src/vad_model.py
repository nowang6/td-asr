"""VAD Model for Voice Activity Detection"""

import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import List, Tuple, Optional
from .utils import load_yaml_config, load_cmvn, apply_cmvn
from .frontend import WavFrontendOnline


class FSMNVADOnline:
    """FSMN VAD model for streaming voice activity detection
    
    Args:
        model_dir: Path to VAD model directory
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
        
        # Create frontend for VAD (lfr_m=5, lfr_n=1 for 400-dim features)
        frontend_conf = self.config.get('frontend_conf', {})
        self.frontend = WavFrontendOnline(
            fs=frontend_conf.get('fs', 16000),
            window=frontend_conf.get('window', 'hamming'),
            n_mels=frontend_conf.get('n_mels', 80),
            frame_length=frontend_conf.get('frame_length', 25),
            frame_shift=frontend_conf.get('frame_shift', 10),
            dither=frontend_conf.get('dither', 0.0),
            lfr_m=frontend_conf.get('lfr_m', 5),
            lfr_n=frontend_conf.get('lfr_n', 1),
        )
        
        # Load ONNX model
        model_file = "model_quant.onnx" if quantize else "model.onnx"
        model_path = self.model_dir / model_file
        
        sess_options = ort.SessionOptions()
        sess_options.inter_op_num_threads = 1
        sess_options.intra_op_num_threads = 1
        
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        
        # Get model input/output names
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # VAD parameters
        model_conf = self.config.get('model_conf', {})
        self.sample_rate = model_conf.get('sample_rate', 16000)
        self.window_size_ms = model_conf.get('window_size_ms', 200)
        self.frame_in_ms = model_conf.get('frame_in_ms', 10)
        self.speech_noise_thres = model_conf.get('speech_noise_thres', 0.6)
        self.max_end_silence_time = model_conf.get('max_end_silence_time', 800)
        self.max_start_silence_time = model_conf.get('max_start_silence_time', 3000)
        self.speech_to_sil_time_thres = model_conf.get('speech_to_sil_time_thres', 150)
        
        # State variables for streaming
        self.reset()
    
    def reset(self):
        """Reset VAD state"""
        self.frontend.reset()
        self.is_speaking = False
        self.speech_start_time = 0
        self.speech_end_time = 0
        self.silence_duration = 0
        self.speech_duration = 0
        self.audio_samples = []
        
    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract features for VAD
        
        Args:
            audio: Audio waveform
            
        Returns:
            Features with shape [n_frames, 400]
        """
        # Extract fbank features
        features = self.frontend.extract_feat(audio)
        
        # Apply CMVN
        if len(features) > 0:
            features = apply_cmvn(features, self.cmvn)
        
        return features
    
    def infer(self, features: np.ndarray) -> np.ndarray:
        """Run VAD inference
        
        Args:
            features: Input features with shape [n_frames, 400]
            
        Returns:
            VAD probabilities with shape [n_frames, 2] (silence, speech)
        """
        if len(features) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        
        # Prepare input
        features_input = features.astype(np.float32)[np.newaxis, :, :]  # [1, T, 400]
        features_length = np.array([features.shape[0]], dtype=np.int32)
        
        # Run inference
        inputs = {
            'speech': features_input,
            'speech_lengths': features_length,
        }
        
        outputs = self.session.run(self.output_names, inputs)
        
        # Output shape: [1, T, 2]
        probs = outputs[0][0]  # [T, 2]
        
        return probs
    
    def process_chunk(self, audio_chunk: np.ndarray) -> Tuple[bool, List[Tuple[int, int]]]:
        """Process audio chunk and return VAD results
        
        Args:
            audio_chunk: Audio chunk (float32 array)
            
        Returns:
            Tuple of (is_speaking, segments)
            - is_speaking: Whether speech is detected
            - segments: List of (start_sample, end_sample) tuples
        """
        # Add to frontend buffer
        self.frontend.add_audio(audio_chunk)
        
        # Extract features
        features, consumed_samples = self.frontend.extract_feat_streaming(return_samples=True)
        
        if len(features) == 0:
            return self.is_speaking, []
        
        # Apply CMVN
        features = apply_cmvn(features, self.cmvn)
        
        # Run inference
        probs = self.infer(features)
        
        # Process VAD results
        segments = []
        for i, prob in enumerate(probs):
            speech_prob = prob[1]  # Probability of speech
            
            if speech_prob > self.speech_noise_thres:
                # Speech detected
                if not self.is_speaking:
                    self.is_speaking = True
                    self.speech_start_time = len(self.audio_samples)
                self.silence_duration = 0
                self.speech_duration += self.frame_in_ms
            else:
                # Silence detected
                if self.is_speaking:
                    self.silence_duration += self.frame_in_ms
                    
                    # Check if silence is long enough to end speech
                    if self.silence_duration >= self.max_end_silence_time:
                        self.is_speaking = False
                        self.speech_end_time = len(self.audio_samples)
                        segments.append((self.speech_start_time, self.speech_end_time))
                        self.speech_duration = 0
                        self.silence_duration = 0
        
        # Store audio samples
        self.audio_samples.extend(audio_chunk.tolist())
        
        return self.is_speaking, segments
    
    def get_speech_segments(self) -> List[Tuple[int, int]]:
        """Get detected speech segments
        
        Returns:
            List of (start_sample, end_sample) tuples
        """
        segments = []
        
        if self.is_speaking:
            # Current speech segment is still active
            self.speech_end_time = len(self.audio_samples)
            segments.append((self.speech_start_time, self.speech_end_time))
        
        return segments

