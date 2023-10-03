# Code repo for Learning Compact Compositional Embeddings via Regularized Pruning for Recommendation (CERP)

This is the PyTorch implementation of our ICDM 2023 [paper](https://arxiv.org/abs/2309.03518):
> Xurong Liang, Tong Chen, Quoc Viet Hung Nguyen, Jianxin Li, and Hongzhi Yin, 
> "Learning Compact Compositional Embeddings via Regularized Pruning for Recommendation,"
> in ICDM, 2023. 

## Dataset

- Both preprocessed Gowalla and Yelp2020 datasets can be downloaded [here](https://drive.google.com/drive/folders/10-gsZtXTCNBGKZgU3s-eYV07sZ7RFfHJ?usp=drive_link). Please place the ***data*** folder under ***~/CERP_code/CERP/***
- Interaction matrix, adjacency graph, train and test samples all gathered by running the data loader from [LightGCN](https://github.com/gusye1234/LightGCN-PyTorch).

---

## Quick Start


- Prune on Gowalla using MLP as base recommender
    ```shell
    user@xxx: ~/CERP_code/CERP$ python3 prune.py --data_type gowalla --sparsity_rates .5 --bucket_size 1000 --model mlp
    ```
- Retrain a particular pruned embedding table obtained above
  ```shell
  user@xxx: ~/CERP_code/CERP$ python3 retrain.py --data_type gowalla --retrain_emb_sparsity .5 --bucket_size 1000 --model mlp
  ```
- Prune on Gowalla using GCN as base recommender
  ```shell
  user@xxx: ~/CERP_code/CERP$ python3 prune.py --data_type gowalla --sparsity_rates .5 --bucket_size 1000 --model gcn
  ```
- To retrain a particular pruned embedding table obtained using LightGCN recommender, we use the **modified LightGCN version (CERP-LightGCN)**
  ```shell
  user@xxx: ~/CERP_code/CERP-LightGCN/code$ python3 main.py --recdim 128 --dataset_name gowalla --CERP_embedding_bucket_size 1000 \
   --path_to_load_CERP_pruned_embs_for_retraining ~/CERP_code/CERP/res/gowalla/GCN/embedding \
  --retrain_sparsity .5 --early_stop 3
  ```

---

## Note

- **Sparsity rate** here is relative to the bucket size and latent dimension size. The relationship 
between sparsity rate (s), bucket size (b), latent dim (d) and target parameter size (t) can be expressed as 
$$t = (1 - s) \times (b \times 2 \times d)$$
- Difference in notation between program and paper:
  
  | notation in paper | notation in program |                       meaning                        |
  |:-----------------:|:-------------------:|:----------------------------------------------------:|
  |         P         |         R_v         |              The meta embedding table P              |
  |       $S_p$       |         R_s         |            The soft threshold matrix of P            |
  |         Q         |         Q_v         |              The meta embedding table Q              |
  |       $S_q$       |         Q_s         |            The soft threshold matrix of Q            |
  |      $\eta$       |          K          | The temperature scalar used in pruning loss equation |


