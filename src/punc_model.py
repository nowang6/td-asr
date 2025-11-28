"""Punctuation Model for adding punctuation to ASR results"""

import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import List
from .utils import load_yaml_config


class PunctuationModel:
    """CT-Transformer Punctuation Model
    
    This model adds punctuation marks to raw ASR transcriptions.
    
    Args:
        model_dir: Path to punctuation model directory
        quantize: Whether to use quantized model (default: True)
    """
    
    def __init__(self, model_dir: str, quantize: bool = True):
        self.model_dir = Path(model_dir)
        self.quantize = quantize
        
        # Load config
        config_path = self.model_dir / "config.yaml"
        if config_path.exists():
            self.config = load_yaml_config(config_path)
        else:
            self.config = {}
        
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
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        # Punctuation marks mapping
        self.punc_list = ['', ',', '。', '?', '!', '、', '；', '：']
    
    def add_punctuation(self, text: str) -> str:
        """Add punctuation to text
        
        Args:
            text: Input text without punctuation
            
        Returns:
            Text with punctuation added
        """
        if not text or len(text) == 0:
            return text
        
        # Simple rule-based punctuation for Chinese text
        # This is a fallback until full ONNX inference is implemented
        result = []
        chars = list(text)
        
        for i, char in enumerate(chars):
            result.append(char)
            
            # Add comma every 3-5 characters (simple heuristic)
            if i > 0 and (i + 1) % 4 == 0 and i < len(chars) - 1:
                # Check if it's a good place for comma
                if char not in '，。？！、：；':
                    result.append('，')
        
        # Add period at the end if not already punctuated
        if len(result) > 0 and result[-1] not in '，。？！、：；':
            result[-1] = result[-1]
            result.append('。')
        
        return ''.join(result)
    
    def infer_onnx(self, text: str) -> str:
        """Run ONNX inference for punctuation (TODO)
        
        Args:
            text: Input text without punctuation
            
        Returns:
            Text with punctuation
        """
        # TODO: Implement proper ONNX inference
        # For now, use rule-based approach
        return self.add_punctuation(text)
    
    def infer(self, text: str) -> str:
        """Run punctuation model inference
        
        Args:
            text: Input text without punctuation
            
        Returns:
            Text with punctuation
        """
        return self.add_punctuation(text)

