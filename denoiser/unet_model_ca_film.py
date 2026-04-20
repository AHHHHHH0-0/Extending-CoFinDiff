import torch
import torch.nn as nn

from config import denoiser_config
from .unet_blocks import EncoderBlock, DecoderBlock, ResBlock, SpatialCrossAttention, FiLM
from .time_steps import timestep_embedding


class UNetDenoiserCAFilm(nn.Module):
    """
    U-Net denoiser with two conditionings:
      - Micro conditions (trend, realized vol) via cross-attention
      - Macro conditions (interest rate, VIX) via FiLM after each cross-attention

    Architecture per level:
      ResBlock(h, t_emb) -> CrossAttention(h, micro_tokens) -> FiLM(h, macro_emb)
    """
    
    def __init__(
        self,
        in_channels: int,
        base_channels: int = denoiser_config.BASE_CHANNELS,
        channel_mult: list = denoiser_config.CHANNEL_MULT,
        num_res_blocks: int = denoiser_config.NUM_RES_BLOCKS,
        time_embed_dim: int = denoiser_config.TIME_EMBED_DIM,
        cond_context_dim: int = denoiser_config.COND_CONTEXT_DIM,
        num_heads: int = denoiser_config.CROSS_ATTN_NUM_HEADS,
        num_macro_scalars: int = denoiser_config.NUM_MACRO_SCALARS,
    ):
        super().__init__()
        
        self.base_channels = base_channels
        
        # Compute channel sizes at each level
        channels = [base_channels * mult for mult in channel_mult]
        
        # Timestep embedding MLP
        self.time_mlp_embed = nn.Sequential(
            nn.Linear(base_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # Initial convolution
        self.input_conv = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)
        
        # Encoder: ResBlocks + CrossAttention + FiLM at each level
        self.encoder_blocks = nn.ModuleList()
        self.encoder_attn_blocks = nn.ModuleList()
        self.encoder_film_blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.encoder_blocks.append(EncoderBlock(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                time_embed_dim=time_embed_dim,
                num_res_blocks=num_res_blocks,
                downsample=True
            ))
            self.encoder_attn_blocks.append(SpatialCrossAttention(
                channels=channels[i + 1],
                context_dim=cond_context_dim,
                num_heads=num_heads
            ))
            self.encoder_film_blocks.append(FiLM(
                num_macro_scalars=num_macro_scalars,
                channels=channels[i + 1],
            ))
        
        # Bottleneck: ResBlock -> CrossAttention -> FiLM -> ResBlock
        self.bottleneck_res1 = ResBlock(channels[-1], channels[-1], time_embed_dim)
        self.bottleneck_attn = SpatialCrossAttention(
            channels=channels[-1],
            context_dim=cond_context_dim,
            num_heads=num_heads
        )
        self.bottleneck_film = FiLM(
            num_macro_scalars=num_macro_scalars,
            channels=channels[-1],
        )
        self.bottleneck_res2 = ResBlock(channels[-1], channels[-1], time_embed_dim)
        
        # Decoder: Upsample + Skip + ResBlocks + CrossAttention + FiLM at each level
        self.decoder_blocks = nn.ModuleList()
        self.decoder_attn_blocks = nn.ModuleList()
        self.decoder_film_blocks = nn.ModuleList()
        for i in range(len(channels) - 1, 0, -1):
            self.decoder_blocks.append(DecoderBlock(
                in_channels=channels[i],
                out_channels=channels[i - 1],
                skip_channels=channels[i],
                time_embed_dim=time_embed_dim,
                num_res_blocks=num_res_blocks,
                upsample=True
            ))
            self.decoder_attn_blocks.append(SpatialCrossAttention(
                channels=channels[i - 1],
                context_dim=cond_context_dim,
                num_heads=num_heads
            ))
            self.decoder_film_blocks.append(FiLM(
                num_macro_scalars=num_macro_scalars,
                channels=channels[i - 1],
            ))
        
        # Output convolution
        self.output_conv = nn.Sequential(
            nn.GroupNorm(min(32, channels[0]), channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], in_channels, kernel_size=3, padding=1)
        )
        
        # Initialize output layer to zero for stable training
        nn.init.zeros_(self.output_conv[-1].weight)
        nn.init.zeros_(self.output_conv[-1].bias)
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        micro_cond_tokens: torch.Tensor,
        macro_emb: torch.Tensor,
    ) -> torch.Tensor:

        # Create timestep embedding
        t_emb = timestep_embedding(t, self.base_channels)  # (B, base_channels)
        t_emb = self.time_mlp_embed(t_emb)  # (B, time_embed_dim)
        
        # Initial convolution
        h = self.input_conv(x)
        
        # Encoder
        skips = []
        for enc, attn, film in zip(
            self.encoder_blocks, self.encoder_attn_blocks, self.encoder_film_blocks
        ):
            h, skip = enc(h, t_emb)
            h = attn(h, micro_cond_tokens)
            h = film(h, macro_emb)
            skips.append(skip)
        
        # Bottleneck
        h = self.bottleneck_res1(h, t_emb)
        h = self.bottleneck_attn(h, micro_cond_tokens)
        h = self.bottleneck_film(h, macro_emb)
        h = self.bottleneck_res2(h, t_emb)
        
        # Decoder
        for dec, attn, film in zip(
            self.decoder_blocks, self.decoder_attn_blocks, self.decoder_film_blocks
        ):
            skip = skips.pop()
            h = dec(h, skip, t_emb)
            h = attn(h, micro_cond_tokens)
            h = film(h, macro_emb)
        
        # Output
        return self.output_conv(h)
    
    def get_num_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
