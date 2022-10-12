import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio, attn_drp, drp):
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(p=attn_drp)
        self.proj = nn.Linear(dim, dim)
        self.block_drop = nn.Dropout(p=drp)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim),
            nn.GELU(),
            nn.Linear(mlp_ratio * dim, dim)
        )
        self.block_norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.block_norm2 = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):

        shape = x.shape[: -1] + (3, self.n_heads, self.head_dim)
        qkv = self.qkv(self.block_norm1(x)).view(shape)  # batch_size, n_patches +label, img_npy, n_heads, heads_dim
        qkv = qkv.permute(2, 0, 3, 1, 4)  # img_npy, batch_size, n_heads, n_patches +label, heads_dim
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (batch_size, n_heads, n_patches +label, heads_dim)
        attn = q @ k.transpose(-2, -1) * self.scale  # (batch_size, n_heads, n_patches +label, n_patches +label)
        attn = self.attn_drop(attn.softmax(dim=-1))  # (batch_size, n_heads, n_patches +label, n_patches +label)
        attn_out = attn @ v  # (batch_size, n_heads, n_patches +label, head_dim)
        attn_out = attn_out.transpose(1, 2)  # (batch_size, n_patches + label, n_heads, head_dim)
        attn_out = attn_out.flatten(2)  # (batch_size, n_patches + label, dim)
        # для визуализации
        #self.attn = attn

        att = self.block_drop(self.proj(attn_out))  # (batch_size, n_patches + label, dim)
        x = x + att
        mlp = self.block_drop(self.mlp(self.block_norm2(x)))  # (batch_size, n_patches + label, hidden_features)
        x = x + mlp

        return x


class VisionTransformer(nn.Module):
    def __init__(self,
                 img_size=8,
                 patch_size=1,
                 n_chan=1024,
                 n_classes=7,
                 dim=768,
                 depth=6,
                 n_heads=12,
                 mlp_ratio=4,
                 pos_1d=True,
                 hybr=False,
                 attn_drp=0.,
                 drp=0.1
                 ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_1d = pos_1d
        self.hybr = hybr
        if self.pos_1d:
            self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.n_patches, dim))
        else:
            n = img_size // patch_size
            self.pos_embed_w = nn.Parameter(torch.zeros(1, dim, 1, n))
            self.pos_embed_h = nn.Parameter(torch.zeros(1, dim, n + 1, 1))
        if not self.hybr:
            self.emb_proj = nn.Conv2d(n_chan,
                                  dim,
                                  kernel_size=patch_size,
                                  stride=patch_size
                                  )
        self.blocks = nn.ModuleList([Block(dim=dim,
                                           n_heads=n_heads,
                                           mlp_ratio=mlp_ratio,
                                           attn_drp=attn_drp,
                                           drp=drp
                                           )
                                     for _ in range(depth)
                                     ])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.output = nn.Linear(dim, n_classes)
        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)
        self.apply(_init)
        nn.init.constant_(self.output.weight, 0)
        nn.init.constant_(self.output.bias, 0)
        if self.pos_1d:
            nn.init.normal_(self.pos_embed, std=0.02)
        else:
            nn.init.normal_(self.pos_embed_w, std=0.02)
            nn.init.normal_(self.pos_embed_h, std=0.02)
        nn.init.constant_(self.cls_token, 0)

    def forward(self, x):
        # получение размера батча
        batch_size = x.shape[0]
        if not self.hybr:
            x = self.emb_proj(x)  # (batch_size, dim, n_patches ** 0.5, n_patches ** 0.5)
        x = x.flatten(2)  # (batch_size, dim, n_patches)
        x = x.transpose(1, 2)  # (batch_size, n_patches, dim)

        cls_token = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, label, dim)
        x = torch.cat((cls_token, x), dim=1)  # (batch_size, label + n_patches, dim)
        if self.pos_1d:
            x = x + self.pos_embed  # (batch_size, label + n_patches, dim)
        else:
            pos_embed_w = self.pos_embed_w.expand(-1, -1, self.img_size // self.patch_size + 1, -1)
            pos_embed_h = self.pos_embed_h.expand(-1, -1, -1, self.img_size // self.patch_size)
            pos = (pos_embed_w + pos_embed_h).flatten(2)[:, :, - (1 + self.n_patches):]
            x = x + pos.permute(0, 2, 1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)  # (batch_size, label + n_patches, dim)
        x = x[:,1:]
        #cls_first = x[:, 0]  # (batch_size, dim)
        # x = self.output(cls_first)
        return x

if __name__ == '__main__':
    from skimage import io
    vit = VisionTransformer(depth=6,
                           n_heads=12,
                           img_size=256,
                           n_chan=2,
                           dim=768,
                           patch_size=256,
                           pos_1d=True,
                           hybr=True,
                           n_classes=1)
    img = io.imread("D:\\Desktop\\1.bmp")
    #rgb = torch.randn([16, 1024, 8, 8])
    out1 = vit(img)
    print(out1.shape)