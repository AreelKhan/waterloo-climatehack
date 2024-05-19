import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

from typing import Dict, Any
from numbers import Number


from scipy.stats import special_ortho_group

import pytorch_lightning as L


class PreprocessNetwork(nn.Module):
    def __init__(self, in_features=38, out_features=64, bias=False):
        super().__init__()
        self.d_in = in_features
        self.d_out = out_features
        self.batch_norm = nn.BatchNorm3d(self.d_in)
        self.linear = nn.Linear(self.d_in, self.d_out, bias=bias)
        with torch.no_grad():
            nn.init.xavier_normal_(self.linear.weight)
        self.layer_norm = nn.LayerNorm(self.d_out, bias=bias)
        
    def forward(self, x):
        """x: (B, C, L, H, W, C)"""
        x = self.batch_norm(x)
        x = x.permute(0, 2, 3, 4, 1)  # (B, L, H, W, C)
        x = self.linear(x)  # C: 38 -> D
        x = self.layer_norm(x)
        return x


class PercevierCrossAttention(nn.Module):
    def __init__(self,
        d_latents=512,
        num_heads=8,
        dropout=0.1,
        ffn_widening_factor=4,
        hidden_activation=nn.SiLU(),
        use_query_residual=True,
        bias=False
    ):
        super().__init__()
        self.use_query_residual = use_query_residual
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_latents,
            num_heads=num_heads,
            dropout=0,  # We do not want to dropout any attention in the first cross attention layer
            batch_first=True,
            bias=bias
        )
        self.ln1 = nn.LayerNorm(d_latents, bias=bias)

        # FFN
        self.linear1 = nn.Linear(d_latents, ffn_widening_factor * d_latents, bias=bias)
        self.linear2 = nn.Linear(ffn_widening_factor * d_latents, d_latents, bias=bias)
        self.act = hidden_activation
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.ln2 = nn.LayerNorm(d_latents, bias=bias)

    def forward(self, inputs, latents):
        z, _ = self.cross_attention(
            query=latents,
            key=inputs,
            value=inputs,
            need_weights=False
        )
        if self.use_query_residual:
            z = latents + z
        z = self.ln1(z)

        # FFN
        y = self.linear1(z)
        y = self.act(y)
        y = self.linear2(y)
        y = self.dropout(y)

        if self.use_query_residual:
            y = y + z
        y = self.ln2(y)
        return y


class Perceiver(nn.Module):
    def __init__(self,
            d_latents=512,
            num_latents=64,
            num_blocks=6,
            num_cross_attention_heads=8,
            num_self_attention_heads=8,
            ffn_widening_factor=4,
            hidden_activation=nn.SiLU(),  # Disable fast path by setting actfn, it breaks with no-bias
            dropout=0.1,
            use_query_residual=True,
            bias=False
        ):
        super().__init__()
        
        latent_embedding = torch.empty((1, num_latents, d_latents), dtype=torch.float32).normal_(0, 1/np.sqrt(d_latents))
        self.latent_embedding = nn.Parameter(latent_embedding)

        self.cross_attention = PercevierCrossAttention(
            d_latents=d_latents,
            num_heads=num_cross_attention_heads,
            dropout=dropout,
            ffn_widening_factor=ffn_widening_factor,
            hidden_activation=hidden_activation,
            use_query_residual=use_query_residual,
            bias=bias
        )

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_latents,
            nhead=num_self_attention_heads,
            dim_feedforward=d_latents * ffn_widening_factor,
            dropout=dropout,
            activation=hidden_activation,
            batch_first=True,
            norm_first=True,
            bias=bias
        )

        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_blocks)

        self.output_layer = nn.Linear(d_latents, 1)  # Always use bias in final layer
    
    def forward(self, inputs, latents):
        batch = inputs.shape[0]
        latents = torch.cat([latents, self.latent_embedding.expand(batch, -1, -1)], dim=1)
        x = self.cross_attention(inputs, latents)
        x = self.encoder(x)
        x = self.output_layer(x).squeeze()
        return x


class SspTransformerEmbedding(nn.Module):
    def __init__(self, input_dim, ssp_dim, h=1, freeze_embeddings=False, bias=False):
        super().__init__()
        self.N = input_dim
        self.D = ssp_dim
        self.freeze_embeddings = freeze_embeddings
        
        n_phases = self.D // 2 + 1
        if self.N == 1:
            theta = torch.randn(n_phases)
            theta[0] = 0
            theta = theta.unsqueeze(-1)
        else:
            basis_generator = special_ortho_group(self.N)
            rot_mats = basis_generator.rvs(n_phases)  # D//2, N, N
            theta = torch.FloatTensor(rot_mats[:, 0, :])
            theta[0] = torch.zeros(self.N, dtype=torch.float32)
        self.theta = nn.Parameter(theta.T, requires_grad=not self.freeze_embeddings)

        if isinstance(h, Number):
            h = torch.ones(input_dim, 1, dtype=torch.float32) * h / np.sqrt(self.N)
        else:
            assert len(h) == self.N
            h = torch.FloatTensor(h).unsqueeze(-1) / np.sqrt(self.N)
        self.register_buffer("h", h)  

        self.linear = nn.Linear(self.D, self.D, bias=bias)
        self.ln = nn.LayerNorm(self.D, bias=bias)

    def get_ssp_vecs(self):
        A = torch.exp(2.j * torch.pi * self.theta / self.h)
        a = torch.fft.irfft(A, n=self.D, dim=-1)
        return a

    def forward(self, x):
        phis = x @ (self.theta / self.h)
        phis = torch.fft.irfft(torch.exp(2.j * torch.pi * phis), n=self.D, dim=-1)
        y = self.linear(phis)
        y = self.ln(y)
        return y


class SspOrtho2D(nn.Module):
    def __init__(self, ssp_dim, h=1, bias=False):
        super().__init__()
        self.N = 2
        self.D = ssp_dim
        n_phases = ssp_dim // 2 + 1
        axes = [
            [1, 0],
            [0, 1],
        ] * ((n_phases - 1) // 2)
        if (n_phases - 1) % 2 == 1:
            axes.append([0, 0])
        axes.insert(0, [0, 0])  # 0 frequency component
        assert len(axes) == n_phases, f"{len(axes)} axes does not match {n_phases} phases"
        axes = torch.FloatTensor(axes)
        self.register_buffer("theta", axes.T)

        if isinstance(h, Number):
            h = torch.ones(self.N, 1, dtype=torch.float32) * h / np.sqrt(self.N)
        else:
            assert len(h) == self.N
            h = torch.FloatTensor(h).unsqueeze(-1) / np.sqrt(self.N)
        self.register_buffer("h", h)
        self.linear = nn.Linear(self.D, self.D, bias=bias)
        self.ln = nn.LayerNorm(self.D, bias=bias)

    def get_ssp_vecs(self):
        A = torch.exp(2.j * torch.pi * self.theta / self.h)
        a = torch.fft.irfft(A, n=self.D, dim=-1)
        return a

    def forward(self, x):
        with torch.no_grad():
            phis = x @ (self.theta / self.h)
            phis = torch.fft.irfft(torch.exp(2.j * torch.pi * phis), n=self.D, dim=-1)
        y = self.linear(phis)
        y = self.ln(y)
        return y


class PVPerceiver(L.LightningModule):
    def __init__(self,
            total_steps, 
            embedding_dim=128, 
            nwp_window=5, 
            num_transformer_layer=12, 
            num_heads=8, 
            lr=5e-4, 
            wd=1e-5,
            init_std=0.02, 
            num_latents=15,
            num_pv_features=3, 
            bias=False, 
            pv_latent=False
        ):
        super().__init__()
        self.total_steps = total_steps
        self.d = embedding_dim
        self.num_transformer_layer = num_transformer_layer
        self.num_heads = num_heads
        self.num_latents = num_latents
        self.pv_latent = pv_latent
        self.num_pv_features = num_pv_features
        self.nwp_window = nwp_window
        self.lr = lr
        self.wd = wd
        self.init_std = init_std
        self.bias = bias
        self.loss_fn = nn.L1Loss()

        self.pv_ssp = SspTransformerEmbedding(1, self.d, bias=self.bias)
        self.time_ssp = SspOrtho2D(self.d, bias=self.bias)
        self.location_ssp = SspTransformerEmbedding(2, self.d, h=0.25, bias=self.bias)
        self.azel_ssp = SspOrtho2D(self.d, bias=self.bias)
        self.static_ssp = SspTransformerEmbedding(3, self.d, bias=self.bias)
        self.pv_features_ssp = SspTransformerEmbedding(self.num_pv_features, self.d, bias=self.bias)

        self.nwp_preprocess = PreprocessNetwork(38, self.d, bias=self.bias)
        self.hrv_preprocess = PreprocessNetwork(1, self.d, bias=self.bias)
        self.perceiver = Perceiver(
            self.d,
            num_latents=self.num_latents,
            num_blocks=self.num_transformer_layer,
            num_cross_attention_heads=self.num_heads,
            num_self_attention_heads=self.num_heads,
            bias=self.bias
        )

        # Init weights
        for module in [self.nwp_preprocess, self.hrv_preprocess, self.pv_ssp, self.time_ssp, self.location_ssp, self.azel_ssp, self.static_ssp, self.pv_features_ssp]:
            module.apply(self._init_embedding_ln)
        self.perceiver.apply(self._init_weights)

    @torch.no_grad()
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.init_std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif hasattr(module, "latent_embedding"):
            module.latent_embedding.data.normal_(mean=0.0, std=self.init_std)

    @torch.no_grad()
    def _init_embedding_ln(self, module):
        if isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                module.bias.data.zero_()
            module.weight.data.fill_(self.init_std)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        preds = self.forward(batch)
        targets = batch["targets"]
        loss = self.loss_fn(preds, targets)
        with torch.no_grad():
            mae = F.l1_loss(preds, targets)
        self.log("train_loss", loss, on_epoch=True)
        self.log("train_mae", mae, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        preds = self.forward(batch)
        targets = batch["targets"]
        mse = F.mse_loss(preds, targets)
        mae = F.l1_loss(preds, targets)
        self.log("val_mse", mse, on_epoch=True)
        self.log("val_mae", mae, on_epoch=True)
    
    def configure_optimizers(self) -> torch.optim.AdamW:
        super().configure_optimizers()
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd, betas=(0.95, 0.99))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            pct_start=0.3,
            final_div_factor=100,
            total_steps=self.total_steps
        )
        lr_scheduler_config = {
            # REQUIRED: The scheduler instance
            "scheduler": scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "step",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
            "monitor": None,
            # If set to `True`, will enforce that the value specified 'monitor'
            # is available when the scheduler is updated, thus stopping
            # training if not found. If set to `False`, it will only produce a warning
            "strict": False,
            # If using the `LearningRateMonitor` callback to monitor the
            # learning rate progress, this keyword can be used to specify
            # a custom logged name
            "name": "OneCycleLR",
        }

        return [optimizer], [lr_scheduler_config]
    
    def train_dataloader(self):
        return super().train_dataloader()
    
    def forward(self, batch) -> torch.Tensor:
        nwp_emb = self.nwp_preprocess(batch["weather"])
        hrv_emb = self.hrv_preprocess(batch["hrv"])

        time_emb = self.time_ssp(batch["time"])  # B, L, D
        weather_time_emb = self.time_ssp(batch["weather_time"])  # B, L, D
        location_emb = self.location_ssp(batch["location"]).unsqueeze(1)  # B, 1, H, W, D
        hrv_location_emb = self.location_ssp(batch["hrv_location"]).unsqueeze(1)  # B, 1, H, W, D
        static_emb = self.static_ssp(batch["static"])  # B, 1, D
        pv_features_emb = self.pv_features_ssp(batch["pv_features"])  # B, 1, D
        azel_emb = self.azel_ssp(batch["azel"])  # B, L, D
        pv_emb = self.pv_ssp(batch["pv"])  # B, L, D

        site_location = location_emb[:, :, self.nwp_window//2, self.nwp_window//2, :]  # B, 1, D

        nwp_emb = nwp_emb + weather_time_emb[:, :, None, None, :] + location_emb
        nwp_emb = nwp_emb.flatten(1, 3)  # B, L*H*W, D

        hrv_emb = hrv_emb + time_emb[:, :12, None, None, :] + hrv_location_emb
        hrv_emb = hrv_emb.flatten(1, 3)  # B, L*H*W, D

        pv_emb = pv_emb + time_emb[:, :12, :] + site_location + azel_emb[:, :12, :]

        future_emb = azel_emb[:, 12:60, :] + time_emb[:, 12:60, :] + site_location
        
        input_seq = torch.cat([
            static_emb,
            pv_features_emb,
            pv_emb,
            nwp_emb,
            hrv_emb
        ], dim=1)
        if self.pv_latent:
            latent_seq = torch.cat([future_emb, static_emb, pv_features_emb, pv_emb], dim=1)
        else:
            latent_seq = torch.cat([future_emb, static_emb], dim=1)
        
        out = self.perceiver(input_seq, latent_seq)
        preds = out[:, :48]
        return preds