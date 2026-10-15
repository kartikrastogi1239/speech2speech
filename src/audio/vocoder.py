"""Spanish Speech Synthesis (TTS) and Vocoder interfaces."""

from typing import Optional, Union
from pathlib import Path
import torch


class SpanishTTSPipeline:
    """Spanish Text-to-Speech synthesizer.

    Integrates Meta's Massively Multilingual Speech (MMS-TTS) for Spanish (es).
    MMS-TTS is a VITS-based model providing natural Spanish speech output.
    """

    def __init__(
        self,
        model_id: str = "facebook/mms-tts-spa",
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._tokenizer = None

    def _lazy_load(self) -> None:
        """Load pretrained model and tokenizer on first synthesis call."""
        if self._model is None or self._tokenizer is None:
            try:
                from transformers import VitsModel, AutoTokenizer

                print(f"[TTS] Loading Spanish TTS model: {self.model_id} onto {self.device}...")
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                self._model = VitsModel.from_pretrained(self.model_id).to(self.device)
                self._model.eval()
            except ImportError:
                raise ImportError(
                    "transformers is required for Spanish TTS. "
                    "Install via `pip install transformers`."
                )

    def synthesize(self, spanish_text: str) -> torch.Tensor:
        """Synthesize Spanish speech waveform from input Spanish text string.

        Args:
            spanish_text: Text string in Spanish.

        Returns:
            torch.Tensor: Audio waveform of shape (num_samples,) at model's sample rate (16000 Hz).
        """
        self._lazy_load()
        inputs = self._tokenizer(spanish_text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = self._model(**inputs)
            waveform = output.waveform[0].cpu()

        return waveform

    def save_wav(self, waveform: torch.Tensor, save_path: Union[str, Path], sample_rate: int = 16000) -> None:
        """Save synthesized waveform to a WAV file."""
        import soundfile as sf

        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), waveform.squeeze().numpy(), sample_rate)
