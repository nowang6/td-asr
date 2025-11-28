"""2-pass ASR Engine"""
import numpy as np
from typing import Optional, Dict, List
from loguru import logger
from pathlib import Path

from .models import VADModel, OnlineASRModel, OfflineASRModel, PUNCModel
from .frontend import WavFrontend, WavFrontendOnline
from .config import (
    VAD_MODEL_DIR,
    ONLINE_ASR_MODEL_DIR,
    OFFLINE_ASR_MODEL_DIR,
    PUNC_MODEL_DIR,
    SAMPLE_RATE,
)

class TwoPassASREngine:
    """2-pass ASR Engine for real-time speech recognition"""
    
    def __init__(
        self,
        vad_model_dir: Optional[Path] = None,
        online_asr_model_dir: Optional[Path] = None,
        offline_asr_model_dir: Optional[Path] = None,
        punc_model_dir: Optional[Path] = None,
        quantize: bool = True,
        thread_num: int = 2,
    ):
        # Model directories
        self.vad_model_dir = vad_model_dir or VAD_MODEL_DIR
        self.online_asr_model_dir = online_asr_model_dir or ONLINE_ASR_MODEL_DIR
        self.offline_asr_model_dir = offline_asr_model_dir or OFFLINE_ASR_MODEL_DIR
        self.punc_model_dir = punc_model_dir or PUNC_MODEL_DIR
        
        logger.info("Loading models...")
        
        # Load models
        self.vad_model = VADModel(self.vad_model_dir, quantize, thread_num)
        self.online_asr_model = OnlineASRModel(self.online_asr_model_dir, quantize, thread_num)
        self.offline_asr_model = OfflineASRModel(self.offline_asr_model_dir, quantize, thread_num)
        self.punc_model = PUNCModel(self.punc_model_dir, quantize, thread_num)
        
        # Frontends
        self.online_frontend = WavFrontendOnline(
            fs=SAMPLE_RATE,
            n_mels=80,
            frame_length=25,
            frame_shift=10,
            lfr_m=7,
            lfr_n=6,
        )
        self.offline_frontend = WavFrontend(
            fs=SAMPLE_RATE,
            n_mels=80,
            frame_length=25,
            frame_shift=10,
            lfr_m=7,
            lfr_n=6,
        )
        
        # Audio buffer for offline processing
        self.audio_buffer = np.array([], dtype=np.float32)
        
        logger.info("Models loaded successfully")
    
    def reset(self):
        """Reset engine state"""
        self.vad_model.reset()
        self.online_asr_model.reset()
        self.online_frontend.reset()
        self.audio_buffer = np.array([], dtype=np.float32)
    
    def process_online(
        self,
        audio_chunk: np.ndarray,
        is_final: bool = False,
    ) -> Dict:
        """
        Process audio chunk in online mode (first pass)
        
        Args:
            audio_chunk: Audio chunk (float32, shape: [T])
            is_final: Whether this is the final chunk
            
        Returns:
            Result dict with:
            - text: Recognized text
            - is_final: Whether result is final
            - is_endpoint: Whether VAD endpoint detected
        """
        # Extract features
        features = self.online_frontend.extract_fbank_online(
            audio_chunk,
            cmvn=self.vad_model.cmvn,
            is_final=is_final,
        )
        
        if len(features) == 0:
            return {
                "text": "",
                "is_final": False,
                "is_endpoint": False,
            }
        
        # VAD
        probs, is_speech, is_endpoint = self.vad_model.infer(features, is_final)
        
        # Only process if speech detected
        if not np.any(is_speech) and not is_final:
            return {
                "text": "",
                "is_final": False,
                "is_endpoint": is_endpoint,
            }
        
        # Online ASR
        text = self.online_asr_model.infer(features, is_final)
        
        # Accumulate audio for offline processing
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
        
        return {
            "text": text,
            "is_final": is_final,
            "is_endpoint": is_endpoint,
        }
    
    def process_offline(self) -> Dict:
        """
        Process accumulated audio in offline mode (second pass)
        
        Returns:
            Result dict with:
            - text: Final recognized text with punctuation
            - is_final: Always True
        """
        if len(self.audio_buffer) == 0:
            return {
                "text": "",
                "is_final": True,
            }
        
        # Extract features
        features = self.offline_frontend.extract_fbank(
            self.audio_buffer,
            cmvn=self.offline_asr_model.cmvn,
        )
        
        if len(features) == 0:
            return {
                "text": "",
                "is_final": True,
            }
        
        # Offline ASR
        text = self.offline_asr_model.infer(features)
        
        # Punctuation
        if text:
            text = self.punc_model.infer(text)
        
        # Clear buffer
        self.audio_buffer = np.array([], dtype=np.float32)
        
        return {
            "text": text,
            "is_final": True,
        }
    
    def process_2pass(
        self,
        audio_chunk: np.ndarray,
        is_final: bool = False,
    ) -> List[Dict]:
        """
        Process audio chunk in 2-pass mode
        
        Args:
            audio_chunk: Audio chunk (float32, shape: [T])
            is_final: Whether this is the final chunk
            
        Returns:
            List of result dicts:
            - Online results (intermediate)
            - Offline result (if endpoint or final)
        """
        results = []
        
        # Online processing
        online_result = self.process_online(audio_chunk, is_final)
        if online_result["text"]:
            results.append(online_result)
        
        # Offline processing if endpoint detected or final
        if online_result["is_endpoint"] or is_final:
            offline_result = self.process_offline()
            if offline_result["text"]:
                results.append(offline_result)
        
        return results

