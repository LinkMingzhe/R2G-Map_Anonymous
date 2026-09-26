"""This file contains the model definition of MaskGen.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License."""
import numpy as np

import math

import torch

import torch.nn as nn

from einops import rearrange

import json

from open_clip.transformer import text_global_pool

from omegaconf import OmegaConf

from pathlib import Path

from diffusers.models.attention import JointTransformerBlock

from diffusers.models.normalization import AdaLayerNormContinuous

from modeling.modules import BaseModel

from modeling.modules.blocks import WeightTiedLMHead

from huggingface_hub import PyTorchModelHubMixin

def get_masking_ratio(progress, mode = "arccos") -> torch.Tensor:
    """ Get masking ratio. """
    if not isinstance(progress, torch.Tensor):
        r = torch.tensor(progress)
    else:
        r = progress
    if mode == "root":
        val_to_mask = 1 - (r ** 0.5)
    elif mode == "square":
        val_to_mask = 1 - (r ** 2)
    elif mode == "cosine":
        val_to_mask = torch.cos(r * math.pi * 0.5)
    elif mode == "arccos":
        val_to_mask = torch.acos(r) / (math.pi * 0.5)
    elif mode == "linear":
        val_to_mask = 1 - r
    else:
        raise ValueError("Invalid mode. Choose between 'linear','square', 'cosine', 'arccos', 'root'.")
    return val_to_mask

def open_clip_text_encoding(clip_tokenizer, clip_encoder, text):
    idxs = clip_tokenizer(text).to(clip_encoder.token_embedding.weight.device)
    cast_dtype = clip_encoder.transformer.get_cast_dtype()
    x = clip_encoder.token_embedding(idxs).to(cast_dtype)  # [batch_size, n_ctx, d_model]

    x = x + clip_encoder.positional_embedding.to(cast_dtype)

    for block in clip_encoder.transformer.resblocks[:-1]:
        x = block(x, attn_mask=clip_encoder.attn_mask)
    x_penultimate = x
    x = clip_encoder.transformer.resblocks[-1](x_penultimate, attn_mask=clip_encoder.attn_mask)

    x = clip_encoder.ln_final(x)  # [batch_size, n_ctx, transformer.width]

    pooled_embed, x = text_global_pool(x, idxs, clip_encoder.text_pool_type)
    pooled_embed = pooled_embed @ clip_encoder.text_projection
    pooled_embed = pooled_embed.unsqueeze(1)

    return x_penultimate, pooled_embed

class MaskGen_VQ(BaseModel, PyTorchModelHubMixin, tags=["arxiv:2501.07730", "text-to-image-generation"], repo_url="https://github.com/bytedance/1d-tokenizer", license="apache-2.0"):
    def __init__(self, config):
        if isinstance(config, dict):
            config = OmegaConf.create(config)

        super().__init__()
        image_seq_len = config.model.vq_model.num_latent_tokens
        target_codebook_size = config.model.vq_model.codebook_size
        condition_num_classes = config.model.maskgen.condition_num_classes
        embed_dim = config.model.maskgen.decoder_embed_dim
        depth = config.model.maskgen.decoder_depth
        num_heads = config.model.maskgen.decoder_num_heads

        self.text_embed_dim = config.model.vq_model.get("text_embed_dim", 768)
        self.micro_condition = config.model.maskgen.micro_condition
        self.micro_condition_embed_dim = config.model.maskgen.micro_condition_embed_dim
        self.sample_aesthetic_score = config.model.maskgen.get("sample_aesthetic_score", 6.0)
        self.text_drop_prob = config.model.maskgen.text_drop_prob

        self.text_embed_proj = nn.Linear(
            self.text_embed_dim,
            embed_dim
        )
        if self.micro_condition:
            self.cond_pooled_proj = nn.Linear(
                self.text_embed_dim + self.micro_condition_embed_dim, embed_dim 
            )
        else:
            self.cond_pooled_proj = nn.Linear(
                self.text_embed_dim, embed_dim
            )

        self.blocks = nn.ModuleList([
            JointTransformerBlock(
                dim=embed_dim,
                num_attention_heads=num_heads,
                # diffusers==0.29.2 interprets attention_head_dim as the full
                # output width and internally divides by num_attention_heads.
                # Passing embed_dim keeps the block output at embed_dim.
                attention_head_dim=embed_dim,
                context_pre_only=d==(depth-1)
            ) for d in range(depth)])

        self.norm = AdaLayerNormContinuous(embed_dim, embed_dim, elementwise_affine=False, eps=1e-6)

        self.embeddings = nn.Embedding(target_codebook_size + 1 + condition_num_classes + 1, embed_dim)  # one additional token for masking, keep unused 1001 for compatibility

        self.pos_embed = nn.init.trunc_normal_(nn.Parameter(torch.zeros(1, image_seq_len, embed_dim)), 0., 0.02)

        if config.model.maskgen.get("weight_tying", True):
            self.lm_head = WeightTiedLMHead(self.embeddings, target_codebook_size)
        else:
            self.lm_head = nn.Linear(embed_dim, target_codebook_size, bias=True)

        self.condition_num_classes = condition_num_classes
        self.image_seq_len = image_seq_len
        self.mask_token_id = target_codebook_size
        self.target_codebook_size = target_codebook_size
        self.none_condition_id = self.condition_num_classes + self.target_codebook_size + 1
        self.mask_schedule_strategy = config.model.maskgen.get("mask_schedule_strategy", "arccos")
        
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if (isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d)):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                module.bias.data.zero_()
            if module.weight is not None:
                module.weight.data.fill_(1.0)
        elif isinstance(module, (AdaLayerNormContinuous)):
            module.linear.weight.data.zero_()
            module.linear.bias.data.zero_()

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights and config to a local directory."""
        # Assume 'self.config' is your DictConfig object
        # Convert to a regular dictionary
        dict_config = OmegaConf.to_container(self.config)
        # Save as JSON
        file_path = Path(save_directory) / "config.json"
        with open(file_path, 'w') as json_file:
            json.dump(dict_config, json_file, indent=4)
        super()._save_pretrained(save_directory)

    def masking_input_tokens(self, input_tokens):
        batch_size, seq_len = input_tokens.shape
        device = input_tokens.device

        timesteps = torch.zeros((batch_size,), device=device).float().uniform_(0, 1.0)
        mask_ratio = get_masking_ratio(timesteps, self.mask_schedule_strategy)
        mask_ratio = torch.clamp(mask_ratio, min=1e-6, max=1.)
        num_token_masked = (seq_len * mask_ratio).round().clamp(min=1)
        batch_randperm = torch.rand(batch_size, seq_len, device=device).argsort(dim=-1)
        masks = batch_randperm < rearrange(num_token_masked, 'b -> b 1')
        masked_tokens = torch.where(masks, self.mask_token_id, input_tokens)
        return masked_tokens, masks

    def preprocess_condition(
        self, 
        condition, 
        clip_tokenizer,
        clip_encoder,
    ):
        # In this case, the condition is a list of strings
        # By default, we assume using open-clip for text encoding
        condition = condition + [""] # add null embedding
        condition, condition_pooled = open_clip_text_encoding(clip_tokenizer, clip_encoder, condition)
        # set condition to null embedding
        drop_label_mask = (torch.rand((condition.shape[0] - 1, 1, 1), dtype=torch.float) < self.text_drop_prob).to(condition)
        condition = condition[:-1] * (1.0 - drop_label_mask) + condition[-1:] * drop_label_mask
        
        condition_pooled = condition_pooled[:-1] * (1.0 - drop_label_mask) + condition_pooled[-1:] * drop_label_mask
        
        return condition, condition_pooled

    def get_sinusoidal_encoding(
        self,
        timesteps: torch.Tensor,
        scale: float = 1,
        max_period: int = 1000,
    ):
        """
        from diffusers
        """
        assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"
        embedding_dim = self.micro_condition_embed_dim
        half_dim = embedding_dim // 2
        exponent = -math.log(max_period) * torch.arange(
            start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
        )
        exponent = exponent / (half_dim)

        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]

        # scale embeddings
        emb = scale * emb

        # concat sine and cosine embeddings
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        # zero pad
        if embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb
        

    def concat_micro_cond(
        self,
        condition,
        aesthetic_score,
    ):
        conds = [condition.squeeze(1)]
        conds.append(self.get_sinusoidal_encoding(aesthetic_score*100))
        conds = torch.cat(conds, dim=-1).unsqueeze(1)

        return conds

    def forward(
        self, 
        input_tokens, 
        condition,
        condition_pooled,
        aesthetic_score=None,
    ):
        # Token space:
        #  [0, codebook_size - 1]                       : those are the learned quantized image tokens
        #  codebook_size                                : the mask token used to mask image tokens
        #  [codebook_size + 1, codebook_size + nclass]  : the imagenet class tokens
        #  codebook_size + 1 + nclass                   : the class drop label
        # prepend condition token
        if self.training:
            input_ids, masks = self.masking_input_tokens(input_tokens)
        else:
            input_ids = input_tokens
            masks = None

        embeddings = self.embeddings(input_ids)

        # linear proj to ensure a same number channel
        condition = self.text_embed_proj(condition)

        if self.micro_condition:
            condition_pooled = self.concat_micro_cond(condition_pooled, aesthetic_score)
        condition_pooled = self.cond_pooled_proj(condition_pooled)

        x = embeddings
        x = x + self.pos_embed[:, :x.shape[1]]

        for blk in self.blocks:
            condition, x = blk(x, condition, condition_pooled.squeeze(1))
        
        x = self.norm(x, condition_pooled.squeeze(1))

        return self.lm_head(x), masks
    
    # ref: https://github.com/baaivision/MUSE-Pytorch/blob/master/libs/muse.py#L40
    @torch.no_grad()
    def generate(
        self, 
        captions,
        guidance_scale=12.0,
        randomize_temperature=1.5,
        sample_aesthetic_score=None,
        softmax_temperature_annealing=True,
        num_sample_steps=16,
        guidance_decay="cosine",
        guidance_decay_scale_pow=1.0,
        clip_tokenizer=None,
        clip_encoder=None,
        prob_sorting=True,
    ):
        assert guidance_decay in ["linear", "cosine", "none", "flippedcosine"]

        condition, condition_pooled = open_clip_text_encoding(clip_tokenizer, clip_encoder, captions)
        none_cond, none_cond_pooled = open_clip_text_encoding(clip_tokenizer, clip_encoder, [""])
        num_samples = condition.shape[0]
        device = condition.device
        none_cond = none_cond.repeat(num_samples, 1, 1)
        none_cond_pooled = none_cond_pooled.repeat(num_samples, 1, 1)

        ids = torch.full((num_samples, self.image_seq_len), self.mask_token_id, device=device)
        cfg_scale = guidance_scale if guidance_decay == "none" else 0.
        if sample_aesthetic_score is not None:
            sample_aesthetic_score = torch.full((num_samples*2,), self.sample_aesthetic_score, device=device)

        # Add gumbel noise
        def log(t, eps=1e-20):
            return torch.log(t.clamp(min=eps))
        def gumbel_noise(t):
            noise = torch.zeros_like(t).uniform_(0, 1)
            return -log(-log(noise))
        def add_gumbel_noise(t, temperature):
            return t + temperature * gumbel_noise(t)

        for step in range(num_sample_steps):
            ratio = 1. * (step + 1) / num_sample_steps
            annealed_temp = randomize_temperature * (1.0 - ratio)
            is_mask = (ids == self.mask_token_id)

            if guidance_decay == "cosine":
                # ref: https://github.com/sail-sg/MDT/blob/441d6a1d49781dbca22b708bbd9ed81e9e3bdee4/masked_diffusion/models.py#L513C13-L513C23
                scale_pow = torch.ones((1), device=device) * guidance_decay_scale_pow
                scale_step = (1 - torch.cos(
                    (ratio ** scale_pow) * torch.pi)) * 1/2
                cfg_scale = (guidance_scale - 1) * scale_step + 1
            elif guidance_decay == "flippedcosine":
                scale_pow = torch.ones((1), device=device) * guidance_decay_scale_pow
                scale_step = (torch.cos(
                    (ratio ** scale_pow) * torch.pi)) * 1/2
                cfg_scale = (guidance_scale - 1) * scale_step + 1
            elif guidance_decay == "linear":
                cfg_scale = ratio * (guidance_scale - 1) + 1

            if cfg_scale != 0:
                logits = self.forward(
                    torch.cat([ids, ids], dim=0),
                    torch.cat([condition, none_cond], dim=0),
                    torch.cat([condition_pooled, none_cond_pooled], dim=0),
                    aesthetic_score=sample_aesthetic_score,
                )[0]
                cond_logits, uncond_logits = logits[:num_samples], logits[num_samples:]
                logits = cond_logits + (cond_logits - uncond_logits) * cfg_scale
            else:
                logits = self.forward(
                    ids, condition, condition_pooled, aesthetic_score=None
                )[0]
            if softmax_temperature_annealing:
                softmax_temperature = 0.5 + 0.8 * (1 - ratio)
            else:
                softmax_temperature = annealed_temp
            logits = logits / softmax_temperature
            
            prob_ids = logits
            sampled_ids = add_gumbel_noise(prob_ids, annealed_temp).argmax(dim=-1)

            sampled_logits = torch.squeeze(
                torch.gather(logits, dim=-1, index=torch.unsqueeze(sampled_ids, -1)), -1)
            sampled_ids = torch.where(is_mask, sampled_ids, ids)
            sampled_logits = torch.where(is_mask, sampled_logits, +np.inf).float()

            # masking
            mask_ratio = get_masking_ratio(ratio, self.mask_schedule_strategy)
            mask_len = torch.floor(self.image_seq_len * mask_ratio).to(device)
            mask_len = torch.maximum(
                torch.Tensor([1]).to(device),
                torch.minimum(torch.sum(is_mask, dim=-1, keepdims=True) - 1, mask_len)
            )[0].squeeze()
            
            if prob_sorting:
                confidence = add_gumbel_noise(sampled_logits, annealed_temp) # How sorting works with gumbel noise? -> sampling without replacement 
            else:
                confidence = sampled_logits

            sorted_confidence, _ = torch.sort(confidence, axis=-1)
            cut_off = sorted_confidence[:, mask_len.long() - 1:mask_len.long()]
            masking = (confidence <= cut_off)
            if step == num_sample_steps - 1:
                ids = sampled_ids
            else:
                ids = torch.where(masking, self.mask_token_id, sampled_ids)

        return ids
