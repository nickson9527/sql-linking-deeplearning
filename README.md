# Integrating Machine Learning Techniques with SQL for Time-Series Data Analysis
### Database Management System – From SQL to NoSQL 

## Abstract
This paper presents a comparative study on the integration of machine learning techniques and SQL for time-series data analysis. The study explores feature selection using genetic algorithms (GA), three types of Recurrent Neural Networks (RNNs), a Transformer model, and the utilization of SQL queries for time-series data retrieval. The results demonstrate the effectiveness of combining these techniques, providing valuable insights into model performance, feature selection impact, and the benefits of SQL integration. This research showcases the potential for comprehensive and efficient time-series data analysis by combining machine learning with SQL.


## Environment
|Library|version|
| -----| -----|
|python|3.8.16|
|torch| 1.12.1+cu113|
|mysql-connector-python| 8.0.33|


## Code usage
### Reproduce
```
Source ./run.sh
```
### train_dl_ga.py
#### Parameters
|Parameter|description|
| -----| -----|
|`--exp_name`| Name of this experiment|
|`--do_train`| add this args to train (must add)|
|`--do_test`| add this args to test (recommand add)|
|`--ga`| add this args to run GA (if not add, it will use `--code` to train one model)|
|`--code`| if no GA, this args will be used to train one model|
|`--patience`| didn't use in our study, leaving it for futher work|
|`--concat_n`| size of sliding window (>=1)|
|`--m_type`| choose from `rnn, lstm, gru, transformer`|
|`--bit_range`| don't change if you do not know what it is|
|`--max_node`| |

```
python3 ./train_dl_ga.py \
    --exp_name transformer_8_512_64_0.5_minmax_concat_$i \
    --do_train \
    --do_test \
    --ga \
    --code 0 1 0 0 0 0 1 \
    --batch_size 128 \
    --num_epoch 500 \
    --patience 1000 \
    --concat_n $i \
    --lr 0.0003 \
    --hidden_layers 8 \
    --hidden_dim 512 \
    --nhead 64 \
    --dropout 0.5 \
    --m_type "transformer" \
    --max_node 7 \
    --bit_range 2 \
    --num_chrom 20 \
    --num_iter 50 \
    --rate_cross 0.8 \
    --rate_mutate 0.3 \
    --seed 6666 
```

### train_ml.py
|Parameter|description|
| -----| -----|
|`--exp_name`| Name of this experiment|
|`--do_train`| add this args to train (must add)|
|`--do_test`| add this args to test (recommand add)|
|`--ga`| add this args to run GA (if not add, it will use `--code` to train one model)|
|`--code`| if no GA, this args will be used to train one model|
|`--concat_n`| size of sliding window (>=1)|
|`--bit_range`| don't change if you do not know what it is|
|`--max_node`| |
```
python3 ./train_ml.py \
    --exp_name linear_minmax_concat_$i \
    --do_train \
    --do_test \
    --ga \
    --minmax \
    --code 0 1 0 0 0 0 1 \
    --concat_n $i \
    --max_node 7 \
    --bit_range 2 \
    --num_chrom 20 \
    --num_iter 50 \
    --rate_cross 0.8 \
    --rate_mutate 0.3 \
    --seed 6666 
```

## Other file
`data.py` load dataset into SQL and query it 
`ga.py` algorithm of Genetic Algorithm
`model.py` including model of RNN, LSTM, GRU, Transformer Encoder Layer