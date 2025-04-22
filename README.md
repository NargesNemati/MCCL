- Original dataset source: https://jmcauley.ucsd.edu/data/amazon/
- Set the desired dataset in train_list.yaml.
 
- To run the MCCL, IGMC, autoencoder(One part of the contrastive learning model), or attention(One part of the contrastive learning model)  models, configure your chosen model in common_config.yaml.  

- To run KGMC by keybert model, select the model in the common_config.yaml file and uncomment the following lines: 
  item_cooc_edge_df: item_keybert
  user_cooc_edge_df: user_keybert
  user_item_cooc_edge_df: user_item_keybert 

- Finally, execute train.py to start training.
