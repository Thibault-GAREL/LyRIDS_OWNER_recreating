"""Modèle de Mention Detection : encoder PLM + tête de classification 3 classes."""
import torch
from torch import nn
from transformers import AutoModel


class MdBioModel(nn.Module):
    """Tagger BIO : pour chaque token, prédit O (0), B (1) ou I (2).

    Le type d'entité est ignoré ici — on ne fait QUE de la détection de mention.
    Le typage est fait par un second modèle (Phase 4).
    """

    def __init__(self, plm_name: str):
        super().__init__()
        self.plm = AutoModel.from_pretrained(plm_name, return_dict=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.plm.config.hidden_size, 3),
        )

    def forward(
        self,
        input_ids: torch.Tensor,        # [batch, max_len]
        attention_mask: torch.Tensor,   # [batch, max_len]
    ) -> torch.Tensor:                  # -> [batch, max_len, 3]
        token_embeddings = self.plm(
            input_ids=input_ids, attention_mask=attention_mask,
        ).last_hidden_state
        return self.classifier(token_embeddings)
