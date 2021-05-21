import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from sgan.models import Encoder, Decoder, get_noise
from .drawer import TrajectoryDrawer
from .attention import ImageAttentionLayer

class CNNTrajectoryGenerator(nn.Module):
    def __init__(self, obs_len=12, pred_len=8, embedding_dim=64, encoder_h_dim=32, decoder_h_dim=32,
                 mlp_dim=64, num_layers=1, dropout=0.5, batch_norm=True,
                 noise_dim=(0,), noise_type='gaussian',
                 noise_mix_type='ped'):
        super(CNNTrajectoryGenerator, self).__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.embedding_dim = embedding_dim
        self.batch_norm = batch_norm
        self.image_drawer = TrajectoryDrawer(target_size=[224, 224])

        self.image_feature_extractor = self._make_image_feature_extractor()
        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=encoder_h_dim,
            dropout=0,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
        )
        self.num_layers = 1
        self.decoder_h_dim = decoder_h_dim
        self.encoder_h_dim = encoder_h_dim
        self.noise_mix_type = noise_mix_type
        self.noise_dim = noise_dim
        self.noise_type = noise_type
        self.dropout = dropout
        self.attentions = nn.ModuleList(
            [ImageAttentionLayer(hidden_dim=encoder_h_dim, image_dim=512) for i in range(2)])
        self.to_spatial = nn.Linear(encoder_h_dim, 2 * pred_len)

    #         self.decoder = Decoder(
    #             pred_len,
    #             embedding_dim=embedding_dim,
    #             h_dim=decoder_h_dim,
    #             mlp_dim=mlp_dim,
    #             num_layers=num_layers,
    #             pool_every_timestep=pool_every_timestep,
    #             dropout=dropout,
    #             bottleneck_dim=bottleneck_dim,
    #             activation=activation,
    #             batch_norm=batch_norm,
    #             pooling_type=pooling_type,
    #             grid_size=grid_size,
    #             neighborhood_size=neighborhood_size,)
    def _make_image_feature_extractor(self):
        resnet = models.resnet18()
        conv1 = nn.Conv2d(11, 64, kernel_size=7, stride=2, padding=3,
                          bias=False)
        module_list = list(resnet.children())
        image_feature_extractor = nn.Sequential(conv1, *module_list[1:-2])

        return image_feature_extractor

    def add_noise(self, _input, seq_start_end, user_noise=None):
        """
        Inputs:
        - _input: Tensor of shape (_, decoder_h_dim - noise_first_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Outputs:
        - decoder_h: Tensor of shape (_, decoder_h_dim)
        """
        if not self.noise_dim:
            return _input

        if self.noise_mix_type == 'global':
            noise_shape = (seq_start_end.size(0),) + self.noise_dim
        else:
            noise_shape = (_input.size(0),) + self.noise_dim

        if user_noise is not None:
            z_decoder = user_noise
        else:
            z_decoder = get_noise(noise_shape, self.noise_type)

        if self.noise_mix_type == 'global':
            _list = []
            for idx, (start, end) in enumerate(seq_start_end):
                start = start.item()
                end = end.item()
                _vec = z_decoder[idx].view(1, -1)
                _to_cat = _vec.repeat(end - start, 1)
                _list.append(torch.cat([_input[start:end], _to_cat], dim=1))
            decoder_h = torch.cat(_list, dim=0)
            return decoder_h

        decoder_h = torch.cat([_input, z_decoder], dim=1)

        return decoder_h

    def forward(self, obs_traj, obs_traj_rel, seq_start_end, obs_team, obs_pos, user_noise=None, images=None):
        batch_size = obs_traj_rel.size(1) // 11
        if images == None:
            images = self.image_drawer.generate_batch_images(obs_traj.cpu())
            images = images.cuda()
            images.requires_grad = False
        images = images.view(batch_size, 11, 224, 224)
        image_features = self.image_feature_extractor(images)
        image_features = torch.squeeze(image_features)
        final_encoder_h = self.encoder(obs_traj_rel)
        hiddens = torch.squeeze(final_encoder_h)
        hiddens = hiddens.view(batch_size, 11, -1)

        for attention_layer in self.attentions:
            _, h = attention_layer(image_features, hiddens)
        spatial = self.to_spatial(h).view(batch_size * 11, -1)
        spatial = spatial.view(-1, self.pred_len, 2)
        spatial = spatial.permute(1, 0, 2)
        #         mlp_decoder_context_input = final_encoder_h.view(
        #                 -1, self.encoder_h_dim)
        #         noise_input = mlp_decoder_context_input
        #         decoder_h = self.add_noise(
        #             noise_input, seq_start_end, user_noise=user_noise)

        #         decoder_h = torch.unsqueeze(decoder_h, 0)
        #         decoder_c = torch.zeros(
        #             self.num_layers, batch_size, self.decoder_h_dim
        #         ).cuda()

        #         state_tuple = (decoder_h, decoder_c)
        #         last_pos = obs_traj[-1]
        #         last_pos_rel = obs_traj_rel[-1]
        #         # Predict Trajectory

        #         decoder_out = self.decoder(
        #             last_pos,
        #             last_pos_rel,
        #             state_tuple,
        #             seq_start_end,
        #         )

        return spatial