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
        
        # LFR cache for streaming (similar to C++ lfr_splice_cache_)
        self.lfr_splice_cache = []
        self.reserve_waveforms = np.array([], dtype=np.float32)
        
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
    
    def extract_feat_streaming(self, return_samples: bool = False, is_finished: bool = False) -> tuple:
        """Extract features from buffered audio for streaming
        
        This implements the C++ ExtractFeats logic with LFR cache management.
        Simplified version that accumulates enough audio before extracting features.
        
        Args:
            return_samples: If True, return number of consumed samples
            is_finished: Whether this is the final chunk
            
        Returns:
            Tuple of (features, consumed_samples)
        """
        # Merge with reserve waveforms if any
        if len(self.reserve_waveforms) > 0:
            self.audio_buffer = np.concatenate([self.reserve_waveforms, self.audio_buffer])
            self.reserve_waveforms = np.array([], dtype=np.float32)
        
        if len(self.audio_buffer) == 0:
            if is_finished and len(self.lfr_splice_cache) > 0:
                # Process remaining cache on final chunk
                features = self.apply_lfr(np.array(self.lfr_splice_cache))
                self.lfr_splice_cache = []
                return features, 0
            return np.zeros((0, self.n_mels * self.lfr_m), dtype=np.float32), 0
        
        # Extract fbank features from current buffer
        fbank_feats = self.extract_fbank(self.audio_buffer)
        
        if len(fbank_feats) == 0:
            return np.zeros((0, self.n_mels * self.lfr_m), dtype=np.float32), 0
        
        # Initialize LFR cache if empty (pad with first frame)
        if len(self.lfr_splice_cache) == 0:
            pad_frames = (self.lfr_m - 1) // 2
            if len(fbank_feats) > 0:
                self.lfr_splice_cache = [fbank_feats[0].copy() for _ in range(pad_frames)]
        
        # Check if we have enough frames to process LFR
        total_frames = len(fbank_feats) + len(self.lfr_splice_cache)
        
        if total_frames >= self.lfr_m or is_finished:
            # Merge cache with new features
            all_fbank = np.array(self.lfr_splice_cache + fbank_feats.tolist())
            
            # Apply LFR (handles padding for final chunk)
            if is_finished:
                # For final chunk, use apply_lfr which handles padding
                lfr_features = self.apply_lfr(all_fbank)
                consumed_samples = len(self.audio_buffer)
                self.audio_buffer = np.array([], dtype=np.float32)
                self.lfr_splice_cache = []
            else:
                # For streaming, calculate how many LFR frames we can generate
                # LFR formula: T_lrf = ceil((T - (lfr_m - 1) / 2) / lfr_n)
                T = len(all_fbank)
                T_lrf = int(np.ceil((T - (self.lfr_m - 1) / 2) / self.lfr_n))
                
                if T_lrf > 0:
                    # Generate LFR features
                    lfr_features = []
                    for i in range(T_lrf):
                        start = i * self.lfr_n
                        end = start + self.lfr_m
                        if end <= T:
                            # Concatenate frames
                            concat_frame = all_fbank[start:end].flatten()
                            lfr_features.append(concat_frame)
                    
                    if len(lfr_features) > 0:
                        lfr_features = np.array(lfr_features, dtype=np.float32)
                        
                        # Calculate frame index for cache (last processed frame)
                        lfr_splice_frame_idx = min(T - 1, (T_lrf - 1) * self.lfr_n + self.lfr_m)
                        
                        # Update cache with remaining frames
                        if lfr_splice_frame_idx < T:
                            self.lfr_splice_cache = all_fbank[lfr_splice_frame_idx:].tolist()
                        else:
                            self.lfr_splice_cache = []
                        
                        # Calculate consumed samples (approximate)
                        # Each LFR frame consumes lfr_n frames
                        consumed_frames = T_lrf * self.lfr_n
                        consumed_samples = (consumed_frames - 1) * self.frame_shift_samples + self.frame_len_samples
                        consumed_samples = min(consumed_samples, len(self.audio_buffer))
                        
                        # Reserve waveforms for overlap
                        reserve_start = max(0, self.frame_len_samples - self.frame_shift_samples)
                        if consumed_samples < len(self.audio_buffer):
                            self.reserve_waveforms = self.audio_buffer[reserve_start:consumed_samples]
                        self.audio_buffer = self.audio_buffer[consumed_samples:]
                    else:
                        # Not enough frames yet, cache them
                        self.lfr_splice_cache.extend(fbank_feats.tolist())
                        self.reserve_waveforms = self.audio_buffer[max(0, self.frame_len_samples - self.frame_shift_samples):]
                        self.audio_buffer = np.array([], dtype=np.float32)
                        lfr_features = np.zeros((0, self.n_mels * self.lfr_m), dtype=np.float32)
                        consumed_samples = 0
                else:
                    # Not enough frames yet, cache them
                    self.lfr_splice_cache.extend(fbank_feats.tolist())
                    self.reserve_waveforms = self.audio_buffer[max(0, self.frame_len_samples - self.frame_shift_samples):]
                    self.audio_buffer = np.array([], dtype=np.float32)
                    lfr_features = np.zeros((0, self.n_mels * self.lfr_m), dtype=np.float32)
                    consumed_samples = 0
            
            if return_samples:
                return lfr_features, consumed_samples
            return lfr_features, 0
        else:
            # Not enough frames, cache them
            self.lfr_splice_cache.extend(fbank_feats.tolist())
            # Reserve waveforms for next iteration
            reserve_start = max(0, self.frame_len_samples - self.frame_shift_samples)
            self.reserve_waveforms = self.audio_buffer[reserve_start:]
            self.audio_buffer = np.array([], dtype=np.float32)
            return np.zeros((0, self.n_mels * self.lfr_m), dtype=np.float32), 0
    
    def reset(self) -> None:
        """Reset audio buffer and caches"""
        self.audio_buffer = np.array([], dtype=np.float32)
        self.lfr_splice_cache = []
        self.reserve_waveforms = np.array([], dtype=np.float32)

