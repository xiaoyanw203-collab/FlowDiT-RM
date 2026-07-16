import torch
import torch.nn as nn
import numpy as np
import math

# ---------------------------------------------------------
# Module 1: core components (reused time embedding and adaLN-Zero)
# ---------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= (embed_dim / 2.)
    omega = 1. / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.freq_dim = frequency_embedding_size

    def forward(self, t):
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.freq_dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return self.mlp(embedding)

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

# ---------------------------------------------------------
# Module 2: StandardDiTBlock 
# ---------------------------------------------------------
class StandardDiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )
        
        # Only six adaLN modulation parameters are needed (self-attention and MLP)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, global_cond):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(global_cond).chunk(6, dim=1)
        
        attn_input = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.self_attn(attn_input, attn_input, attn_input)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        mlp_input = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(mlp_input)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        
        return x

# ---------------------------------------------------------
# Module 3: FlowDiT model (concatenated spatial condition)
# ---------------------------------------------------------
class AblationCondDiT(nn.Module):
    def __init__(
        self,
        input_size=128,
        patch_size=4,
        out_channels=3,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0
    ):
        super().__init__()
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.grid_size = input_size // patch_size

        # Core change 1: channel concatenation
        # x_t(3) + x_0_fspl(3) + cond_spatial(4) = 10 Channels
        in_channels = 3 + 3 + 4
        self.x_embedder = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size) 
        
        # Keep global conditions (time t and frequency f)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.f_embedder = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        num_patches = self.grid_size ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(hidden_size, self.grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        self.blocks = nn.ModuleList([
            StandardDiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)
        ])
        
        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6),
            nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        )
        
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.constant_(self.final_layer[1].weight, 0)
        nn.init.constant_(self.final_layer[1].bias, 0)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.patch_size
        h = w = self.grid_size
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x_t, t, x_0_fspl, cond_spatial, freq):
        x_input = torch.cat([x_t, x_0_fspl, cond_spatial], dim=1) # (B, 10, H, W)
        
        x = self.x_embedder(x_input).flatten(2).transpose(1, 2)
        x = x + self.pos_embed 
    
        # Global control features (t and f)
        t_c = self.t_embedder(t)
        f_c = self.f_embedder(freq)
        global_cond = t_c + f_c 
    
        for block in self.blocks:
            # cond_kv is no longer passed
            x = block(x, global_cond) 
        
        x = self.final_layer(x)
        return self.unpatchify(x)

# Test code
if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device="cpu"
    print(f"Using device: {device}")
    
    model = AblationCondDiT(input_size=128, depth=28).to(device)
    
    # Import the existing flow-matching wrapper for compatibility.
    from flowdit_rm.physics.phys_dit import RectifiedFlowWrapper
    rf_pipeline = RectifiedFlowWrapper(model)
    
    B = 2
    x_1 = torch.randn(B, 3, 128, 128).to(device)      
    x_0 = torch.randn(B, 3, 128, 128).to(device)      
    cond = torch.randn(B, 4, 128, 128).to(device)     
    freq = torch.tensor([[0.8], [-0.5]], dtype=torch.float32).to(device)
    
    loss = rf_pipeline.get_train_loss(x_1, x_0, cond, freq)
    print(f"✅Loss: {loss.item():.4f}")