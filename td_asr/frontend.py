"""Audio frontend for feature extraction"""
import numpy as np
from typing import Optional
import librosa
from scipy import signal

class WavFrontend:
    """Frontend for extracting Fbank features"""
    
    def __init__(
        self,
        fs: int = 16000,
        window: str = "hamming",
        n_mels: int = 80,
        frame_length: int = 25,
        frame_shift: int = 10,
        lfr_m: int = 7,
        lfr_n: int = 6,
        dither: float = 0.0,
    ):
        self.fs = fs
        self.window = window
        self.n_mels = n_mels
        self.frame_length_ms = frame_length
        self.frame_shift_ms = frame_shift
        self.lfr_m = lfr_m
        self.lfr_n = lfr_n
        self.dither = dither
        
        # Calculate frame parameters
        self.frame_length = int(frame_length * fs / 1000)
        self.frame_shift = int(frame_shift * fs / 1000)
        
        # Create mel filterbank
        self.mel_filters = librosa.filters.mel(
            sr=fs,
            n_fft=self.frame_length,
            n_mels=n_mels,
            fmin=0,
            fmax=fs // 2,
        )
        
    def extract_fbank(
        self,
        audio: np.ndarray,
        cmvn: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Extract Fbank features from audio
        
        Args:
            audio: Audio signal (float32, shape: [T])
            cmvn: CMVN statistics for normalization (optional)
            
        Returns:
            Fbank features (shape: [T', n_mels])
        """
        # Add dither
        if self.dither > 0:
            audio = audio + np.random.randn(len(audio)) * self.dither
        
        # Pre-emphasis
        audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
        
        # Frame extraction with windowing
        frames = []
        for i in range(0, len(audio) - self.frame_length + 1, self.frame_shift):
            frame = audio[i:i + self.frame_length]
            
            # Apply window
            if self.window == "hamming":
                window = np.hamming(self.frame_length)
            elif self.window == "hann":
                window = np.hanning(self.frame_length)
            else:
                window = np.ones(self.frame_length)
            
            frame = frame * window
            frames.append(frame)
        
        if not frames:
            return np.zeros((1, self.n_mels), dtype=np.float32)
        
        frames = np.array(frames)
        
        # FFT
        fft = np.fft.rfft(frames, n=self.frame_length)
        magnitude = np.abs(fft)
        power = magnitude ** 2
        
        # Mel filterbank
        mel_spectrum = np.dot(power, self.mel_filters.T)
        
        # Log
        mel_spectrum = np.log(np.maximum(mel_spectrum, 1e-10))
        
        # LFR (Low Frame Rate) - stack and subsample
        if self.lfr_m > 1:
            mel_spectrum = self._apply_lfr(mel_spectrum)
        
        # CMVN normalization
        if cmvn is not None:
            mean = cmvn[0]
            var = cmvn[1]
            mel_spectrum = (mel_spectrum - mean) / np.maximum(np.sqrt(var), 1e-10)
        
        return mel_spectrum.astype(np.float32)
    
    def _apply_lfr(self, features: np.ndarray) -> np.ndarray:
        """Apply Low Frame Rate (LFR) processing"""
        # Stack consecutive frames
        stacked = []
        for i in range(0, len(features) - self.lfr_m + 1, self.lfr_n):
            frame_stack = features[i:i + self.lfr_m].flatten()
            stacked.append(frame_stack)
        
        if not stacked:
            # Pad if necessary
            pad_frames = np.zeros((self.lfr_m - len(features), features.shape[1]))
            stacked = [np.concatenate([features, pad_frames]).flatten()]
        
        return np.array(stacked)

class WavFrontendOnline(WavFrontend):
    """Online frontend that maintains state"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset()
    
    def reset(self):
        """Reset internal state"""
        self.audio_buffer = np.array([], dtype=np.float32)
    
    def extract_fbank_online(
        self,
        audio_chunk: np.ndarray,
        cmvn: Optional[np.ndarray] = None,
        is_final: bool = False,
    ) -> np.ndarray:
        """
        Extract Fbank features from audio chunk (online mode)
        
        Args:
            audio_chunk: New audio chunk (float32, shape: [T])
            cmvn: CMVN statistics
            is_final: Whether this is the final chunk
            
        Returns:
            Fbank features for the new chunk
        """
        # Append to buffer
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
        
        # Extract features
        features = self.extract_fbank(self.audio_buffer, cmvn)
        
        # For online mode, we need to return only new features
        # This is a simplified version - in practice, we'd track what's been processed
        if is_final:
            result = features
            self.reset()
        else:
            # Return features for the chunk (simplified)
            result = features[-len(audio_chunk) // self.frame_shift:]
            # Keep some overlap in buffer
            keep_samples = self.frame_length
            if len(self.audio_buffer) > keep_samples:
                self.audio_buffer = self.audio_buffer[-keep_samples:]
        
        return result

