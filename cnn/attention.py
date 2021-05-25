import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class AttentionModule(nn.Module):
    """
    Query: Image
    Key-Value: Hidden States
    """

    def __init__(self, image_dim, hidden_dim, n_head=16, key_dim=16, value_dim=16, dropout=0.5, len_agents=23):
        super(AttentionModule, self).__init__()

        self.image_dim = image_dim
        self.hidden_dim = hidden_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.n_head = n_head
        self.len_agents = len_agents
        self.w_qs = nn.Linear(hidden_dim, n_head * key_dim, bias=False)
        self.w_ks = nn.Linear(image_dim, n_head * key_dim, bias=False)
        self.w_vs = nn.Linear(image_dim, n_head * value_dim, bias=False)
        self.fc = nn.Linear(n_head * value_dim, hidden_dim, bias=False)
        self.dot_attention = ScaledDotProductAttention(temperature=key_dim ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)

    def forward(self, images, hiddens):
        n_head = self.n_head
        key_dim = self.key_dim
        value_dim = self.value_dim
        batch_size = images.size(0)
        image_dim = images.size(1)
        k = images.view(batch_size, image_dim, -1)
        k = k.permute(0, 2, 1)
        len_image = k.size(1)

        q = hiddens
        residual = q
        len_hidden = self.len_agents
        v = k
        residual = hiddens
        q = self.w_qs(q).view(batch_size, -1, n_head, key_dim)
        k = self.w_ks(k).view(batch_size, len_image, n_head, key_dim)
        v = self.w_vs(v).view(batch_size, len_image, n_head, value_dim)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q, attn = self.dot_attention(q, k, v)
        q = q.transpose(1, 2).contiguous().view(batch_size, self.len_agents, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)
        return q


class PostAttentionFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in=32, d_hid=128, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class ImageAttentionLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, image_dim, hidden_dim, d_inner=128, n_head=16, key_dim=16, value_dim=16, dropout=0.5):
        super(ImageAttentionLayer, self).__init__()
        self.attn = AttentionModule(image_dim=image_dim, hidden_dim=hidden_dim, n_head=n_head, key_dim=key_dim,
                                    value_dim=value_dim, dropout=dropout)
        self.post_ffn = PostAttentionFeedForward(hidden_dim, d_inner, dropout=dropout)

    def forward(self, images, hiddens):
        attn_output = self.attn(
            images, hiddens)
        layer_output = self.post_ffn(attn_output)
        return images, layer_output
