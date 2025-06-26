from typing import Optional

import torch
from torch import nn
from transformers import SiglipVisionConfig
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
from transformers.models.siglip.modeling_siglip import (
    SiglipEncoder,
    SiglipMultiheadAttentionPoolingHead,
    SiglipVisionEmbeddings,
    SiglipVisionModel,
)
from transformers.utils import auto_docstring, can_return_tuple


class SiglipVisionModelNoEmbeddings(SiglipVisionModel):
    config_class = SiglipVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: SiglipVisionConfig):
        super().__init__(config)

        self.vision_model = SiglipVisionTransformer(config)

        # Initialize weights and apply final processing
        self.post_init()

    # def get_input_embeddings(self) -> nn.Module:
    #     return self.vision_model.embeddings.patch_embedding

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        hidden_states,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ) -> BaseModelOutputWithPooling:
        return self.vision_model(
            hidden_states=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.use_head = (
            True if not hasattr(config, "vision_use_head") else config.vision_use_head
        )
        if self.use_head:
            self.head = SiglipMultiheadAttentionPoolingHead(config)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        hidden_states,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = False,
    ) -> BaseModelOutputWithPooling:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.post_layernorm(last_hidden_state)

        pooler_output = self.head(last_hidden_state) if self.use_head else None

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


def load_vit(path_vit_pt: str):
    vision_config = SiglipVisionConfig(name_or_path="google/siglip-base-patch16-224")
    model = SiglipVisionModelNoEmbeddings(vision_config)

    vit_weights = torch.load(path_vit_pt)
    encoder_weights = vit_weights["encoder"]
    head_weights = vit_weights["head"]
    post_layernorm_weights = vit_weights["post_layernorm"]

    model.vision_model.encoder.load_state_dict(encoder_weights)
    model.vision_model.head.load_state_dict(head_weights)
    model.vision_model.post_layernorm.load_state_dict(post_layernorm_weights)
    model = model.to("cuda")

    return model


def load_embedder(path_embeddings_pt: str):
    vision_config = SiglipVisionConfig(name_or_path="google/siglip-base-patch16-224")
    model = SiglipVisionEmbeddings(vision_config)

    weights = torch.load(path_embeddings_pt)

    model.load_state_dict(weights)
    model = model.to("cuda")

    return model
