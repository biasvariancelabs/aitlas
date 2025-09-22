from torch import nn


class Norm2d(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class FPNHead(nn.Module):
    def __init__(self, embed_dim, share_weights=False) -> None:
        super().__init__()
        self.share_weights = share_weights
        if self.share_weights:
            self.fpn1 = nn.Sequential(
                Norm2d(embed_dim),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )
            self.do_fpn1 = lambda x: self.fpn1(self.fpn2(x))
        else:
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                Norm2d(embed_dim),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )

        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2)
        )

        # self.fpn3 = nn.Identity()

        # self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """
        InputL B X C X H X W
        """
        features = []
        if self.share_weights:
            ops = [
                self.do_fpn1,
                self.fpn2,
                # self.fpn3, self.fpn4
            ]
        else:
            ops = [
                self.fpn1,
                self.fpn2,
                # self.fpn3, self.fpn4
            ]
        for i in range(len(ops)):
            features.append(ops[i](x))

        return tuple(features)


class HFFB(nn.Module):
    def __init__(self, hidden_dim) -> None:
        super().__init__()
        self.convs = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(
                hidden_dim, hidden_dim // 2, 3, padding=1, groups=hidden_dim // 2
            ),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, 1, padding=0),
        )
        self.residual = nn.Conv2d(hidden_dim, hidden_dim, 1)

    def forward(self, x):
        return self.convs(x) + self.residual(x)


class FCNHead(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_layers, target_dim) -> None:
        super().__init__()
        self.proj = nn.Conv2d(embed_dim, hidden_dim, 1)
        convs = []
        for _ in range(num_layers):
            convs.append(HFFB(hidden_dim))
        self.conv_blocks = nn.Sequential(*convs)
        self.pred = nn.Sequential(
            Norm2d(hidden_dim),
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=4),
            nn.GELU(),
            nn.Conv2d(
                hidden_dim // 2, hidden_dim // 4, 3, padding=1, groups=hidden_dim // 4
            ),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 4, hidden_dim // 2, 1, padding=0),
            nn.GELU(),
            nn.ConvTranspose2d(hidden_dim // 2, 3, kernel_size=2, stride=2),
        )

    def forward(self, xp):
        """
        InputL List[B X C X H X W], FPN features
        """
        out = []
        for x in xp:
            x = self.proj(x)
            out.append(self.pred(self.conv_blocks(x)))

        return out
