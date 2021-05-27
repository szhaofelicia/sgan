python ~/research/sgan/scripts/train_general_teampos.py  \
  --model 'team_pos' \
  --dataset_name 'dota-ti-9.v1' \
  --dataset_dir /media/felicia/Data/sgan_data/ \
  --output_dir /media/felicia/Data/ \
  --dset 'dota' \
  --trajD 2 \
  --delim tab \
  --d_type 'local' \
  --pred_len 8 \
  --encoder_h_dim_g 32 \
  --encoder_h_dim_d 64\
  --decoder_h_dim 32 \
  --embedding_dim 16 \
  --team_embedding_dim 4 \
  --pos_embedding_dim 16 \
  --bottleneck_dim 32 \
  --mlp_dim 128 \
  --num_layers 1 \
  --noise_type gaussian \
  --noise_mix_type global \
  --pool_every_timestep 0 \
  --l2_loss_weight 1 \
  --batch_norm 0 \
  --dropout 0.5 \
  --tp_dropout 0.5 \
  --batch_size 16 \
  --g_learning_rate 1e-3 \
  --g_steps 1 \
  --d_learning_rate 1e-3 \
  --d_steps 2 \
  --checkpoint_every 10 \
  --print_every 50 \
  --num_iterations 40000 \
  --num_epochs 500 \
  --pooling_type 'pool_net' \
  --clipping_threshold_g 1.5 \
  --best_k 10 \
  --g_gamma 1 \
  --d_gamma 1 \
  --interaction_activation attentiontp \
  --checkpoint_name dota.team_pos_attentiontp_v3.6.d5.e16.pe16.te4.tpd5.gg10.dg10.l10 \
  --restore_from_checkpoint 0

    # --dataset_dir /scratch/sz2257/data/ \
  # --output_dir ./dota/results \