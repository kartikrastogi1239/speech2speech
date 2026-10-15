"""Inference decoding routines: Greedy Decoding and Beam Search."""

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.transformer import Seq2SeqTransformer


class SequenceGenerator:
    """Autoregressive text/token generator for Seq2SeqTransformer."""

    def __init__(
        self,
        model: Seq2SeqTransformer,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.device = device or next(model.parameters()).device

    @torch.no_grad()
    def greedy_decode(
        self,
        src: torch.Tensor,
        max_len: int = 128,
        src_mask: Optional[torch.Tensor] = None,
        is_audio: bool = False,
    ) -> torch.Tensor:
        """Autoregressive greedy search decoding.

        Args:
            src: Source sequence tensor of shape (batch_size, src_len) or (batch_size, time, dim)
            max_len: Maximum target generation length
            src_mask: Source padding mask
            is_audio: True if src is acoustic feature frames

        Returns:
            Tensor of generated token IDs: (batch_size, generated_seq_len)
        """
        self.model.eval()
        batch_size = src.size(0)

        # 1. Encode source once into memory
        memory = self.model.encode(src=src, src_mask=src_mask, is_audio=is_audio)

        # 2. Initialize target with BOS token: (batch_size, 1)
        ys = torch.full((batch_size, 1), self.bos_token_id, dtype=torch.long, device=self.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for _ in range(max_len):
            tgt_len = ys.size(1)
            causal_mask = Seq2SeqTransformer.generate_square_subsequent_mask(tgt_len, device=self.device)

            # Decode target sequence so far
            out = self.model.decode(
                tgt=ys,
                memory=memory,
                tgt_mask=causal_mask,
                memory_mask=src_mask,
            )
            # Project the very last token to vocabulary logits: (batch_size, tgt_vocab_size)
            logits = self.model.output_head(out[:, -1, :])
            next_tokens = torch.argmax(logits, dim=-1, keepdim=True)

            # Update finished status if token is EOS
            finished = finished | (next_tokens.squeeze(1) == self.eos_token_id)

            # Append predicted token
            ys = torch.cat([ys, next_tokens], dim=1)

            # Terminate if all sequences in batch reached EOS
            if finished.all():
                break

        return ys

    @torch.no_grad()
    def beam_search_decode(
        self,
        src: torch.Tensor,
        beam_size: int = 4,
        max_len: int = 128,
        alpha: float = 0.6,
        src_mask: Optional[torch.Tensor] = None,
        is_audio: bool = False,
    ) -> List[List[int]]:
        """Beam search decoding with length penalty.

        Args:
            src: Single sample source tensor (1, src_len) or (1, time, dim)
            beam_size: Number of beams
            max_len: Maximum generation steps
            alpha: Length penalty factor: ((5 + len) / 6)^alpha

        Returns:
            List of token IDs representing top hypothesis.
        """
        self.model.eval()
        if src.size(0) != 1:
            raise ValueError("Beam search currently implemented for batch_size=1.")

        memory = self.model.encode(src=src, src_mask=src_mask, is_audio=is_audio)

        # Hypotheses: list of tuples (token_ids, cumulative_log_prob)
        beams = [([self.bos_token_id], 0.0)]
        completed = []

        for _ in range(max_len):
            candidates = []
            for seq, score in beams:
                if seq[-1] == self.eos_token_id:
                    completed.append((seq, score))
                    continue

                tgt_tensor = torch.tensor([seq], dtype=torch.long, device=self.device)
                causal_mask = Seq2SeqTransformer.generate_square_subsequent_mask(
                    len(seq), device=self.device
                )

                out = self.model.decode(
                    tgt=tgt_tensor,
                    memory=memory,
                    tgt_mask=causal_mask,
                    memory_mask=src_mask,
                )
                logits = self.model.output_head(out[:, -1, :])
                log_probs = F.log_softmax(logits, dim=-1).squeeze(0)

                topk_probs, topk_tokens = torch.topk(log_probs, beam_size)
                for prob, tok in zip(topk_probs.tolist(), topk_tokens.tolist()):
                    candidates.append((seq + [tok], score + prob))

            if not candidates:
                break

            # Rank candidates by length-penalized score
            def score_fn(item):
                s, lp = item
                lp_penalty = ((5.0 + len(s)) / 6.0) ** alpha
                return lp / lp_penalty

            candidates.sort(key=score_fn, reverse=True)
            beams = candidates[:beam_size]

            if len(completed) >= beam_size:
                break

        if not completed:
            completed = beams

        completed.sort(key=score_fn, reverse=True)
        return completed[0][0]
