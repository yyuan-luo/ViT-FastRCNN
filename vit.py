import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., linearize_out=True):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.linearize_out = linearize_out
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, out_features=dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        if self.linearize_out == True:
            out = self.to_out(out)
            
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.norm_last = nn.LayerNorm(heads*dim_head)
        self.layers = nn.ModuleList([])
        
        for i in range(depth):
            if i < depth - 1:
                self.layers.append(nn.ModuleList([
                    Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                    FeedForward(dim, mlp_dim, dropout = dropout)
                ]))
            elif i == depth - 1:
                self.layers.append(nn.ModuleList([
                    Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, linearize_out=False),
                    FeedForward(heads*dim_head, mlp_dim, dropout = dropout)
                ]))

    def forward(self, x):
        for i, (attn, ff) in enumerate(self.layers):
            if i < len(self.layers) - 1:
                x = attn(x) + x
                x = ff(x) + x
                x = self.norm(x)
            elif i == len(self.layers) - 1:
                x = attn(x)
                x = ff(x)
                x = self.norm_last(x)

        return x
    
class ResidualBlock(nn.Module):
    def __init__( self, in_channels: int, out_channels: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride= 1,
                     padding=1, groups=1, bias=False, dilation=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.gelu1 = nn.GELU()
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride= 1,
                     padding=1, groups=1, bias=False, dilation=1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.gelu2 = nn.GELU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.norm1(y)
        y = self.gelu1(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = y + x
        y = self.gelu2(y)
        return y

class ViTFeatureExtractor(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'none', channels = 3, dim_head = 768, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'none'}, 'pool type must be either cls (cls token) or none'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()
        
        self.reshape_feature = Rearrange('b (p1 p2) d -> b d p1 p2', p1=image_height // patch_height, p2 = image_width // patch_width)
        self.conv = nn.Conv2d(heads*dim_head, 512, 1)
        self.residual_block = ResidualBlock(in_channels=512, out_channels=512)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)
        x = x[: 0] if self.pool == 'cls' else x[:, 1:]
        x = self.to_latent(x)
        x = self.reshape_feature(x)
        x = self.conv(x)
        x = self.residual_block(x)
        return x
     
     
if __name__ == '__main__':
   img = torch.randn(2, 3, 512, 512) # FIXME: right now, the image has to be a square
   vit = ViTFeatureExtractor(
      image_size = 512,
      patch_size = 32,
      num_classes = 1000,
      dim = 1024,
      depth = 6,
      heads = 12,
      mlp_dim = 2048,
      dropout = 0.1,
      emb_dropout = 0.1,
      pool='none'
   )
   with torch.no_grad():
      print(vit(img).shape)