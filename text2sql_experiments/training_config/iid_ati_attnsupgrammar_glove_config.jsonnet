local local_dir = ""; 
{
  "dataset_reader": {
    "type": "grammar_based_attn_sup",
    "database_file": "",
    "load_cache": true,
    "save_cache": false,
    "schema_path": local_dir +"data/sql data/atis-schema.csv",
    "token_indexers": { "tokens": { "type": "single_id" } }
  },
  "iterator": {
    "type": "bucket",
    "batch_size": 1,
    "padding_noise": 0,
    "sorting_keys": [ [ "tokens", "num_tokens" ] ]
  },
  "model": {
    "type": "attn_sup_grmr",
    "action_embedding_dim": 100,
    "attn_loss_lambda": 0.1,
    "decoder_beam_search": { "beam_size": 5 },
    "dropout": 0.5,
    "encoder": {
      "type": "lstm",
      "bidirectional": true,
      "dropout": 0,
      "hidden_size": 200,
      "input_size": 100,
      "num_layers": 1
    },
    "input_attention": { "type": "dot_product" },
    "max_decoding_steps": 300,
    "mydatabase": "atis",
    "schema_path": local_dir +"data/sql data/atis-schema.csv",
    "utterance_embedder": {
      "tokens": {
        "type": "embedding",
        "embedding_dim": 100,
        "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz",
        "trainable": true,
        "vocab_namespace": "tokens"
      }
    }
  },
  "train_data_path": local_dir +"data/sql data/atis/new_question_split/aligned_train.json",
  "validation_data_path": local_dir +"data/sql data/atis/new_question_split/aligned_final_dev.json",
  "trainer": {
    "cuda_device": 1,
    "learning_rate_scheduler": {
      "type": "noam",
      "model_size": 400,
      "warmup_steps": 800
    },
    "num_epochs": 50,
    "num_serialized_models_to_keep": 1,
    "optimizer": {
      "type": "adam",
      "lr": 0.001
    },
    "patience": 10,
    "validation_metric": "+seq_acc"
  }
}