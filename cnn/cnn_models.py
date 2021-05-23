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
                 noise_dim=(0,), noise_type='gaussian', image_pretrained=False,
                 noise_mix_type='ped', attention_layer_num=2, n_head=16, key_dim=16, value_dim=16, decoder_inner_dim=128):
        super(CNNTrajectoryGenerator, self).__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.embedding_dim = embedding_dim
        self.batch_norm = batch_norm
        self.image_drawer = TrajectoryDrawer(target_size=[224, 224])

        self.image_feature_extractor = self._make_image_feature_extractor(pretrained=image_pretrained)
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
        self.attention_layer_num = attention_layer_num
        self.n_head = n_head
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.debug_mlp = nn.Linear(2*self.obs_len, encoder_h_dim)
        self.attentions = nn.ModuleList(
            [ImageAttentionLayer(d_inner=decoder_inner_dim, key_dim=key_dim, value_dim=value_dim, n_head=n_head, hidden_dim=encoder_h_dim, image_dim=512, dropout=dropout) for i in range(attention_layer_num)])
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
    def _make_image_feature_extractor(self, pretrained=False):
        resnet = models.resnet18(pretrained=pretrained)
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
        batch_size = len(seq_start_end)
        if images == None:
            images = self.image_drawer.generate_batch_images(obs_traj.cpu())
            images = images.cuda()
            images.requires_grad = False
        pad_images = torch.zeros(batch_size, 11, 224, 224).cuda()
        for i, start_end in enumerate(seq_start_end):
            pad_images[i, 0:start_end[1] - start_end[0], :, :] = images[start_end[0]:start_end[1], :, :]
        images = pad_images
        images = images.view(batch_size, 11, 224, 224)
        image_features = self.image_feature_extractor(images)
        image_features = torch.squeeze(image_features)
        traj = obs_traj_rel.permute(1, 0, 2)
        traj = traj.view(traj.size(0), -1)
        final_encoder_h = self.debug_mlp(traj)
        # final_encoder_h = self.encoder(obs_traj_rel)
        hiddens = torch.squeeze(final_encoder_h)

        hiddens_list = []
        l = []
        for i, start_end in enumerate(seq_start_end):
            hiddens_list.append(hiddens[start_end[0]:start_end[1], :])
            l.append(start_end[1] - start_end[0])
        pad_hiddens = torch.nn.utils.rnn.pad_sequence(hiddens_list)
        pad_hiddens = pad_hiddens.permute(1, 0, 2)
        # print(pad_hiddens.size())
        # hiddens = hiddens.view(batch_size, 11, -1)
        # pad_hiddens = torch.rand(batch_size, 11, hiddens.size(-1)).cuda()
        # pad_hiddens = torch.zeros(batch_size, 11, hiddens.size(-1)).cuda()
        # for i, start_end in enumerate(seq_start_end):
        #     pad_hiddens[i, 0:start_end[1] - start_end[0], :] = hiddens[start_end[0]:start_end[1], :]
        h = pad_hiddens
        for attention_layer in self.attentions:
            _, h = attention_layer(image_features, h)
        # packed_h = torch.zeros(hiddens.size(0), h.size(-1)).cuda()
        # print(l)

        batch_boolean = [False for i in range(h.size(0) * h.size(1))]
        for i in range(h.size(0)):
            t = l[i]
            for j in range(t):
                batch_boolean[i * h.size(1) + j] = True
        h = h.view(-1, h.size(-1))
        # print(h.size())
        h = h[batch_boolean]
        # print(h.size())
        packed_h = h
        # packed_h = torch.nn.utils.rnn.pack_padded_sequence(h, l, batch_first=True, enforce_sorted=False)
        # print(packed_h.data.size())
        # print(packed_h.sorted_indices)
        # print(packed_h.batch_sizes)
        # packed_h = packed_h.data
        # for i, start_end in enumerate(seq_start_end):
        #     packed_h[start_end[0]: start_end[1], :] = h[i, 0: start_end[1]-start_end[0], :]
        spatial = self.to_spatial(packed_h).view(hiddens.size(0), -1)
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
