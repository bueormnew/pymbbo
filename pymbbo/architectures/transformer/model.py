import torch
import torch.nn as nn
from pymbbo.architectures.base_arch import BaseArchitecture
from pymbbo.models.registry import register_architecture

@register_architecture("transformer")
class TransformerArchitecture(BaseArchitecture):
    """
    Mini Transformer Decoder / LLM Architecture Plugin for text generation and token benchmarks.
    """
    ARCH_NAME = "transformer"

    def __init__(self, vocab_size: int = 1000, d_model: int = 128, nhead: int = 4, num_layers: int = 2, max_seq_len: int = 2048, **kwargs):
        super().__init__(vocab_size=vocab_size, d_model=d_model, nhead=nhead, num_layers=num_layers, max_seq_len=max_seq_len, **kwargs)
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        out = self.token_embedding(x) + self.pos_embedding(positions)
        out = self.transformer(out)
        logits = self.fc_out(out)
        return logits

    def generate(self, prompt_tokens: torch.Tensor, max_new_tokens: int = 100) -> torch.Tensor:
        """
        Auto-regressive token generation simulation for token scaling benchmarks.
        """
        curr = prompt_tokens.clone()
        for _ in range(max_new_tokens):
            if curr.size(1) >= self.max_seq_len:
                break
            logits = self.forward(curr)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            curr = torch.cat([curr, next_token], dim=1)
        return curr
