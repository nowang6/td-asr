"""Frontend for feature extraction"""

import numpy as np
from scipy import signal
from scipy.fftpack import dct
from typing import Optional


class WavFrontendOnline:
    """Online frontend for extracting fbank features with LFR
    
    This implements the WavFrontendOnline used in FunASR models.
    
    Args:
        fs: Sample rate (default: 16000)
        window: Window type (default: 'hamming')
        n_mels: Number of mel filters (default: 80)
        frame_length: Frame length in ms (default: 25)
        frame_shift: Frame shift in ms (default: 10)
        dither: Dithering constant (default: 0.0)
        lfr_m: LFR merge factor (default: 7)
        lfr_n: LFR skip factor (default: 6)
    """
    
    def __init__(
        self,
        fs: int = 16000,
        window: str = 'hamming',
        n_mels: int = 80,
        frame_length: int = 25,
        frame_shift: int = 10,
        dither: float = 0.0,
        lfr_m: int = 7,
        lfr_n: int = 6,
    ):
        self.fs = fs
        self.window_type = window
        self.n_mels = n_mels
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.dither = dither
        self.lfr_m = lfr_m
        self.lfr_n = lfr_n
        
        # Calculate frame parameters
        self.frame_len_samples = int(frame_length * fs / 1000)
        self.frame_shift_samples = int(frame_shift * fs / 1000)
        self.n_fft = 512  # Standard FFT size
        
        # Create window
        if window == 'hamming':
            self.window = np.hamming(self.frame_len_samples)
        elif window == 'hanning':
            self.window = np.hanning(self.frame_len_samples)
        else:
            self.window = np.ones(self.frame_len_samples)
        
        # Create mel filterbank
        self.mel_filters = self._create_mel_filters()
        
        # Audio buffer for streaming
        self.audio_buffer = np.array([], dtype=np.float32)
        
    def _create_mel_filters(self) -> np.ndarray:
        """Create mel filterbank matrix"""
        # Frequency points
        low_freq = 0
        high_freq = self.fs / 2
        
        # Convert to mel scale
        low_mel = self._hz_to_mel(low_freq)
        high_mel = self._hz_to_mel(high_freq)
        
        # Create mel points
        mel_points = np.linspace(low_mel, high_mel, self.n_mels + 2)
        hz_points = self._mel_to_hz(mel_points)
        
        # Convert to FFT bin numbers
        bin_points = np.floor((self.n_fft + 1) * hz_points / self.fs).astype(int)
        
        # Create filterbank
        filters = np.zeros((self.n_mels, self.n_fft // 2 + 1))
        for i in range(self.n_mels):
            left = bin_points[i]
            center = bin_points[i + 1]
            right = bin_points[i + 2]
            
            # Rising slope
            for j in range(left, center):
                filters[i, j] = (j - left) / (center - left)
            
            # Falling slope
            for j in range(center, right):
                filters[i, j] = (right - j) / (right - center)
        
        return filters
    
    @staticmethod
    def _hz_to_mel(hz: float) -> float:
        """Convert Hz to mel scale"""
        return 2595 * np.log10(1 + hz / 700)
    
    @staticmethod
    def _mel_to_hz(mel: float) -> float:
        """Convert mel scale to Hz"""
        return 700 * (10 ** (mel / 2595) - 1)
    
    def extract_fbank(self, waveform: np.ndarray) -> np.ndarray:
        """Extract fbank features from waveform
        
        Args:
            waveform: Audio waveform, shape [n_samples]
            
        Returns:
            Fbank features, shape [n_frames, n_mels]
        """
        # Add dithering
        if self.dither > 0:
            waveform = waveform + self.dither * np.random.randn(*waveform.shape)
        
        # Calculate number of frames
        n_frames = (len(waveform) - self.frame_len_samples) // self.frame_shift_samples + 1
        if n_frames <= 0:
            return np.zeros((0, self.n_mels), dtype=np.float32)
        
        # Extract frames
        frames = np.zeros((n_frames, self.frame_len_samples))
        for i in range(n_frames):
            start = i * self.frame_shift_samples
            end = start + self.frame_len_samples
            frames[i] = waveform[start:end] * self.window
        
        # Apply FFT
        fft_frames = np.fft.rfft(frames, n=self.n_fft)
        power_spectrum = np.abs(fft_frames) ** 2
        
        # Apply mel filterbank
        mel_spectrum = np.dot(power_spectrum, self.mel_filters.T)
        
        # Convert to log scale
        mel_spectrum = np.maximum(mel_spectrum, 1e-10)
        log_mel = np.log(mel_spectrum)
        
        return log_mel.astype(np.float32)
    
    def apply_lfr(self, features: np.ndarray) -> np.ndarray:
        """Apply Low Frame Rate (LFR) to features
        
        LFR combines multiple frames to reduce temporal resolution.
        
        Args:
            features: Input features, shape [n_frames, n_mels]
            
        Returns:
            LFR features, shape [n_lfr_frames, n_mels * lfr_m]
        """
        if len(features) == 0:
            return np.zeros((0, self.n_mels * self.lfr_m), dtype=np.float32)
        
        n_frames = features.shape[0]
        n_lfr_frames = (n_frames - self.lfr_m) // self.lfr_n + 1
        
        if n_lfr_frames <= 0:
            return np.zeros((0, self.n_mels * self.lfr_m), dtype=np.float32)
        
        lfr_features = []
        for i in range(n_lfr_frames):
            start = i * self.lfr_n
            end = start + self.lfr_m
            if end <= n_frames:
                # Concatenate frames
                concat_frame = features[start:end].flatten()
                lfr_features.append(concat_frame)
        
        if len(lfr_features) == 0:
            return np.zeros((0, self.n_mels * self.lfr_m), dtype=np.float32)
        
        return np.array(lfr_features, dtype=np.float32)
    
    def extract_feat(self, waveform: np.ndarray) -> np.ndarray:
        """Extract features (fbank + LFR)
        
        Args:
            waveform: Audio waveform, shape [n_samples]
            
        Returns:
            Features with LFR applied, shape [n_lfr_frames, n_mels * lfr_m]
        """
        # Extract fbank
        fbank = self.extract_fbank(waveform)
        
        # Apply LFR
        lfr_features = self.apply_lfr(fbank)
        
        return lfr_features
    
    def add_audio(self, audio_chunk: np.ndarray) -> None:
        """Add audio chunk to buffer for streaming
        
        Args:
            audio_chunk: Audio chunk to add
        """
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
    
    def extract_feat_streaming(self, return_samples: bool = False) -> tuple:
        """Extract features from buffered audio for streaming
        
        This extracts features from the buffer and keeps remaining samples
        for the next chunk.
        
        Args:
            return_samples: If True, return number of consumed samples
            
        Returns:
            Tuple of (features, consumed_samples)
        """
        if len(self.audio_buffer) == 0:
            return np.zeros((0, self.n_mels * self.lfr_m), dtype=np.float32), 0
        
        # Calculate how many frames we can extract
        max_samples = len(self.audio_buffer)
        n_frames = (max_samples - self.frame_len_samples) // self.frame_shift_samples + 1
        
        if n_frames <= 0:
            return np.zeros((0, self.n_mels * self.lfr_m), dtype=np.float32), 0
        
        # Calculate how many complete LFR frames we can extract
        n_lfr_frames = (n_frames - self.lfr_m) // self.lfr_n
        if n_lfr_frames <= 0:
            return np.zeros((0, self.n_mels * self.lfr_m), dtype=np.float32), 0
        
        # Calculate samples needed for these LFR frames
        frames_needed = n_lfr_frames * self.lfr_n + self.lfr_m
        samples_needed = (frames_needed - 1) * self.frame_shift_samples + self.frame_len_samples
        
        # Extract features
        waveform = self.audio_buffer[:samples_needed]
        features = self.extract_feat(waveform)
        
        # Calculate consumed samples (for complete LFR frames)
        consumed_samples = n_lfr_frames * self.lfr_n * self.frame_shift_samples
        
        # Keep remaining audio in buffer
        self.audio_buffer = self.audio_buffer[consumed_samples:]
        
        if return_samples:
            return features, consumed_samples
        return features, 0
    
    def reset(self) -> None:
        """Reset audio buffer"""
        self.audio_buffer = np.array([], dtype=np.float32)

