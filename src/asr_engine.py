"""ASR Engine integrating all components for 2-pass recognition"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .vad_model import FSMNVADOnline
from .asr_model import OnlineASRModel, OfflineASRModel
from .punc_model import PunctuationModel
from .utils import bytes_to_float32


@dataclass
class ASRResult:
    """ASR recognition result"""
    text: str
    is_final: bool
    timestamp: int  # in milliseconds
    

class ASREngine:
    """2-pass ASR Engine with VAD, Online ASR, Offline ASR, and Punctuation
    
    This engine follows the architecture in funasr-wss-server-2pass.cpp:
    1. VAD for voice activity detection
    2. Online ASR for real-time streaming results
    3. Offline ASR for final high-quality results
    4. Punctuation for adding punctuation marks
    
    Args:
        model_config: Dictionary containing model paths
        - vad_dir: Path to VAD model
        - online_model_dir: Path to online ASR model
        - offline_model_dir: Path to offline ASR model
        - punc_dir: Path to punctuation model
        - quantize: Whether to use quantized models
    """
    
    def __init__(self, model_config: Dict[str, str]):
        self.model_config = model_config
        
        # Extract config
        vad_dir = model_config.get('vad_dir')
        online_model_dir = model_config.get('online_model_dir')
        offline_model_dir = model_config.get('offline_model_dir')
        punc_dir = model_config.get('punc_dir')
        quantize = model_config.get('quantize', 'true').lower() == 'true'
        
        # Initialize models
        self.vad_model = None
        if vad_dir:
            self.vad_model = FSMNVADOnline(vad_dir, quantize=quantize)
        
        self.online_asr_model = None
        if online_model_dir:
            self.online_asr_model = OnlineASRModel(online_model_dir, quantize=quantize)
        
        self.offline_asr_model = None
        if offline_model_dir:
            self.offline_asr_model = OfflineASRModel(offline_model_dir, quantize=quantize)
        
        self.punc_model = None
        if punc_dir:
            self.punc_model = PunctuationModel(punc_dir, quantize=quantize)
        
        # State variables
        self.audio_buffer = []
        self.is_speaking = False
        self.speech_start_sample = 0
        self.total_samples = 0
        
        # Results cache
        self.online_results = []
        self.offline_results = []
        
    def reset(self):
        """Reset engine state"""
        if self.vad_model:
            self.vad_model.reset()
        if self.online_asr_model:
            self.online_asr_model.reset()
        
        self.audio_buffer = []
        self.is_speaking = False
        self.speech_start_sample = 0
        self.total_samples = 0
        self.online_results = []
        self.offline_results = []
    
    def process_audio_chunk(
        self, 
        audio_chunk: np.ndarray,
        is_finished: bool = False
    ) -> List[ASRResult]:
        """Process audio chunk and return recognition results
        
        This implements 2-pass recognition:
        1. Online ASR for streaming results (partial, is_final=False)
        2. Offline ASR for final high-quality results (is_final=True)
        
        Args:
            audio_chunk: Audio samples (float32 array)
            is_finished: Whether this is the last chunk
            
        Returns:
            List of ASR results
        """
        results = []
        
        # Add to buffer
        if len(audio_chunk) > 0:
            self.audio_buffer.append(audio_chunk)
            self.total_samples += len(audio_chunk)
        
        # For streaming mode (is_finished=False), return online results
        if not is_finished:
            # Online ASR processing (streaming results)
            # Process the new chunk with online model (maintaining state via cache)
            if self.online_asr_model and len(audio_chunk) > 0:
                # Extract features for this chunk only
                features = self.online_asr_model.extract_features(audio_chunk)
                
                # Run online inference (this maintains encoder cache internally)
                if len(features) > 0:
                    text = self.online_asr_model.infer(features)
                    
                    # Log for debugging
                    from loguru import logger
                    logger.debug(f"Online ASR chunk result: '{text}' (chunk: {len(audio_chunk)} samples, features: {features.shape})")
                    
                    if text and text.strip():
                        # Create intermediate result (2pass-online)
                        result = ASRResult(
                            text=text,
                            is_final=False,
                            timestamp=0
                        )
                        results.append(result)
        
        # For final mode (is_finished=True), return offline results
        else:
            # Process all remaining audio with offline model
            if len(self.audio_buffer) > 0:
                complete_audio = np.concatenate(self.audio_buffer)
                
                # Run offline ASR on complete audio
                if self.offline_asr_model:
                    features = self.offline_asr_model.extract_features(complete_audio)
                    text = self.offline_asr_model.infer(features)
                    
                    # Add punctuation
                    if self.punc_model and text:
                        text = self.punc_model.infer(text)
                    
                    if text and text.strip():
                        # Create final result (2pass-offline)
                        result = ASRResult(
                            text=text,
                            is_final=True,
                            timestamp=0
                        )
                        results.append(result)
            
            # Reset for next session
            self.reset()
        
        return results
    
    def process_audio_bytes(
        self,
        audio_bytes: bytes,
        is_finished: bool = False
    ) -> List[ASRResult]:
        """Process audio bytes (PCM 16-bit format)
        
        Args:
            audio_bytes: PCM 16-bit binary data
            is_finished: Whether this is the last chunk
            
        Returns:
            List of ASR results
        """
        # Convert bytes to float32 array
        audio_chunk = bytes_to_float32(audio_bytes)
        
        # Process audio
        return self.process_audio_chunk(audio_chunk, is_finished)


class ASREngineFactory:
    """Factory for creating ASR engine instances
    
    This factory manages shared model configurations and creates
    independent engine instances for each WebSocket connection.
    """
    
    def __init__(self, model_config: Dict[str, str]):
        self.model_config = model_config
    
    def create_engine(self) -> ASREngine:
        """Create a new ASR engine instance
        
        Returns:
            New ASR engine instance
        """
        return ASREngine(self.model_config)

