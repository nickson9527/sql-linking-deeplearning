# python3 ./train_dl.py --batch_size 16 --n_epochs 1000 --patience 1000 --concat_n 20

# python3 ./train_dl_ga.py \
# --do_train \
# --do_test \
# --ga \
# --batch_size 16 \
# --num_epoch 200 \
# --patience 1000 \
# --concat_n 5 \
# --hidden_layers 4 \
# --hidden_dim 512 \
# --dropout 0.3 \
# --m_type "lstm" \
# --max_node 7 \
# --bit_range 2 \
# --num_chrom 20 \
# --num_iter 100 \
# --rate_cross 0.8 \
# --rate_mutate 0.3 \
# --seed 6666 

python3 ./train_dl_ga.py \
--do_train \
--do_test \
--code 1 1 1 1 1 1 1 \
--batch_size 16 \
--num_epoch 1000 \
--patience 1000 \
--concat_n 1 \
--lr 0.0003 \
--hidden_layers 64 \
--hidden_dim 512 \
--dropout 0.5 \
--m_type "transformer" \
--max_node 7 \
--bit_range 2 \
--num_chrom 20 \
--num_iter 100 \
--rate_cross 0.8 \
--rate_mutate 0.3 \
--seed 6666 