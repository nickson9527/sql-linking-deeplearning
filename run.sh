# python3 ./train_dl.py --batch_size 16 --n_epochs 1000 --patience 1000 --concat_n 20

python3 ./train_dl_ga.py \
    --exp_name test \
    --do_train \
    --do_test \
    --code 1 1 1 1 1 1 1 \
    --batch_size 128 \
    --num_epoch 1000 \
    --patience 1000 \
    --concat_n 20 \
    --lr 0.0003 \
    --hidden_layers 8 \
    --hidden_dim 512 \
    --nhead 64 \
    --dropout 0.5 \
    --m_type "gru" \
    --max_node 7 \
    --bit_range 2 \
    --num_chrom 10 \
    --num_iter 5 \
    --rate_cross 0.8 \
    --rate_mutate 0.3 \
    --seed 6666 

#     --ga \

# for i in 1 5 10 20
# do
#     ### rnn_8_512_0.5
#     python3 ./train_dl_ga.py \
#     --exp_name rnn_8_512_0.5_minmax_concat_$i \
#     --do_train \
#     --do_test \
#     --ga \
#     --code 1 1 1 1 1 1 1 \
#     --batch_size 128 \
#     --num_epoch 500 \
#     --patience 1000 \
#     --concat_n $i \
#     --lr 0.0003 \
#     --hidden_layers 8 \
#     --hidden_dim 512 \
#     --nhead 64 \
#     --dropout 0.5 \
#     --m_type "rnn" \
#     --max_node 7 \
#     --bit_range 2 \
#     --num_chrom 20 \
#     --num_iter 50 \
#     --rate_cross 0.8 \
#     --rate_mutate 0.3 \
#     --seed 6666 
# # done
#     ### lstm_8_512_0.5
#     python3 ./train_dl_ga.py \
#     --exp_name lstm_8_512_0.5_minmax_concat_$i \
#     --do_train \
#     --do_test \
#     --ga \
#     --code 1 1 1 1 1 1 1 \
#     --batch_size 128 \
#     --num_epoch 500 \
#     --patience 1000 \
#     --concat_n $i \
#     --lr 0.0003 \
#     --hidden_layers 8 \
#     --hidden_dim 512 \
#     --nhead 64 \
#     --dropout 0.5 \
#     --m_type "lstm" \
#     --max_node 7 \
#     --bit_range 2 \
#     --num_chrom 20 \
#     --num_iter 50 \
#     --rate_cross 0.8 \
#     --rate_mutate 0.3 \
#     --seed 6666 

# #     ### gru_8_512_0.5
#     python3 ./train_dl_ga.py \
#     --exp_name gru_8_512_0.5_minmax_concat_$i \
#     --do_train \
#     --do_test \
#     --ga \
#     --code 1 1 1 1 1 1 1 \
#     --batch_size 128 \
#     --num_epoch 500 \
#     --patience 1000 \
#     --concat_n $i \
#     --lr 0.0003 \
#     --hidden_layers 8 \
#     --hidden_dim 512 \
#     --nhead 64 \
#     --dropout 0.5 \
#     --m_type "gru" \
#     --max_node 7 \
#     --bit_range 2 \
#     --num_chrom 20 \
#     --num_iter 50 \
#     --rate_cross 0.8 \
#     --rate_mutate 0.3 \
#     --seed 6666 

#     #### transformer_8_512_64_0.5
#     python3 ./train_dl_ga.py \
#     --exp_name transformer_8_512_64_0.5_minmax_concat_$i \
#     --do_train \
#     --do_test \
#     --ga \
#     --code 0 1 0 0 0 0 1 \
#     --batch_size 128 \
#     --num_epoch 500 \
#     --patience 1000 \
#     --concat_n $i \
#     --lr 0.0003 \
#     --hidden_layers 8 \
#     --hidden_dim 512 \
#     --nhead 64 \
#     --dropout 0.5 \
#     --m_type "transformer" \
#     --max_node 7 \
#     --bit_range 2 \
#     --num_chrom 20 \
#     --num_iter 50 \
#     --rate_cross 0.8 \
#     --rate_mutate 0.3 \
#     --seed 6666 
# done

# python3 ./train_dl_ga.py \
#     --exp_name transformer_8_512_64_0.5_concat_10 \
#     --do_train \
#     --do_test \
#     --ga \
#     --code 0 1 0 0 0 0 1 \
#     --batch_size 128 \
#     --num_epoch 500 \
#     --patience 1000 \
#     --concat_n 10 \
#     --lr 0.0003 \
#     --hidden_layers 8 \
#     --hidden_dim 512 \
#     --nhead 64 \
#     --dropout 0.5 \
#     --m_type "transformer" \
#     --max_node 7 \
#     --bit_range 2 \
#     --num_chrom 20 \
#     --num_iter 50 \
#     --rate_cross 0.8 \
#     --rate_mutate 0.3 \
#     --seed 6666 

# for i in 1 5 10 20
# do
#     ### linear
#     python3 ./train_ml.py \
#     --exp_name linear_minmax_concat_$i \
#     --do_train \
#     --do_test \
#     --ga \
#     --code 0 1 0 0 0 0 1 \
#     --concat_n $i \
#     --max_node 7 \
#     --bit_range 2 \
#     --num_chrom 20 \
#     --num_iter 50 \
#     --rate_cross 0.8 \
#     --rate_mutate 0.3 \
#     --seed 6666 
# done