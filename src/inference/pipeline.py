"""End-to-end Hindi-to-Spanish Speech-to-Speech translation pipeline."""

from typing import Any, Dict, Optional, Union
from pathlib import Path
import torch

from src.models.transformer import Seq2SeqTransformer
from src.audio.feature_extractor import MelSpectrogramExtractor
from src.audio.vocoder import SpanishTTSPipeline
from src.data.tokenizer import SimpleCharSubwordTokenizer
from src.inference.translator import SequenceGenerator


class SpeechToSpeechPipeline:
    """End-to-end Hindi Speech -> Spanish Speech Translation Pipeline."""

    def __init__(
        self,
        transformer_model: Seq2SeqTransformer,
        tgt_tokenizer: SimpleCharSubwordTokenizer,
        src_tokenizer: Optional[SimpleCharSubwordTokenizer] = None,
        feature_extractor: Optional[MelSpectrogramExtractor] = None,
        tts_pipeline: Optional[SpanishTTSPipeline] = None,
        device: Optional[torch.device] = None,
    ):
        self.device = device or next(transformer_model.parameters()).device
        self.model = transformer_model.to(self.device)
        self.tgt_tokenizer = tgt_tokenizer
        self.src_tokenizer = src_tokenizer
        self.feature_extractor = feature_extractor or MelSpectrogramExtractor()
        self.tts = tts_pipeline or SpanishTTSPipeline(device=self.device)
        self.generator = SequenceGenerator(
            model=self.model,
            bos_token_id=self.tgt_tokenizer.bos_id,
            eos_token_id=self.tgt_tokenizer.eos_id,
            pad_token_id=self.tgt_tokenizer.pad_id,
            device=self.device,
        )

    def translate_speech(
        self,
        hindi_audio: Union[str, Path, torch.Tensor],
        beam_size: int = 1,
        synthesize_audio: bool = True,
    ) -> Dict[str, Any]:
        """Translate Hindi audio to Spanish text and Spanish speech.

        Args:
            hindi_audio: Path to audio file or raw 1D waveform tensor.
            beam_size: 1 for greedy decoding, >1 for beam search.
            synthesize_audio: Whether to synthesize output Spanish speech waveform.

        Returns:
            dict containing:
                - 'spanish_text': Translated Spanish string
                - 'spanish_audio': Synthesized waveform tensor (if synthesize_audio=True)
                - 'sample_rate': Audio sampling rate
        """
        # 1. Load audio if path provided
        if isinstance(hindi_audio, (str, Path)):
            import soundfile as sf
            wav, _ = sf.read(str(hindi_audio))
            waveform = torch.tensor(wav, dtype=torch.float32)
        else:
            waveform = hindi_audio.float()

        if waveform.dim() == 2:
            waveform = waveform.mean(dim=0)

        # 2. Extract acoustic features: (1, time, n_mels)
        features = self.feature_extractor(waveform.unsqueeze(0)).to(self.device)

        # 3. Autoregressive translation using our custom Transformer
        if beam_size > 1:
            token_ids = self.generator.beam_search_decode(
                src=features, beam_size=beam_size, is_audio=True
            )
        else:
            token_ids_tensor = self.generator.greedy_decode(src=features, is_audio=True)
            token_ids = token_ids_tensor[0].tolist()

        # 4. Detokenize into Spanish string
        spanish_text = self.tgt_tokenizer.decode(token_ids, skip_special_tokens=True)

        result = {
            "spanish_text": spanish_text,
            "spanish_audio": None,
            "sample_rate": 16000,
        }

        # 5. Synthesize Spanish speech if requested
        if synthesize_audio and len(spanish_text.strip()) > 0:
            spanish_waveform = self.tts.synthesize(spanish_text)
            result["spanish_audio"] = spanish_waveform

        return result
