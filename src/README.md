## Configuration

The main entry-point script is ``nd-slicer/src/run.py``. It has the following:

1. Required arguments:

    | Argument         | Description |
    | :--------------: | :---- |
    | ``--data_dir``   | Path to inputs/data caching |
    | ``--output_dir`` | Output directory for writing model predictions and checkpoints |
    | ``--encoder``    | Encoder in Seq2Seq framework |
    | ``--decoder``    | Decoder in Seq2Seq framework |
    | ``--use_pointer``| Whether to use selective pointer networks |


2. Experiment arguments:
                        
    | Argument                | Default                      | Description |
    | :---------------------: | :--------------------------: | :---- |
    | ``--do_train``          |  False                       | Whether to run training |
    | ``--do_eval``           |  False                       | Whether to run evaluation |
    | ``--do_eval_base``      |  False                       | Whether to run baseline evaluation |
    | ``--do_eval_qual``      |  False                       | Whether to run qualitative evaluation |
    | ``--do_eval_loop``      |  False                       | Whether to run qualitative evaluation on loops |
    | ``--do_eval_im``        |  False                       | Whether to run inter-method evaluation |
    | ``--do_eval_crash``     |  False                       | Whether to evaluate crash detection |
    | ``--do_eval_partial``   |  False                       | Whether to run variable aliasing on dev set |
    | ``--do_predict``        |  False                       | Whether to evaluate for partial programs |
    | ``--print_stats``       |  False                       | Print dataset statistics |
    | ``--load_model_path``   |  None                        | Path to trained model: Should contain the .bin files |
    | ``--dataset``           |  False                       | Dataset for intrinsic evaluation |
    | ``--config_name``       |  None                        | Pretrained config name or path if not the same as model_name_or_path |


3. Optional arguments:

    | Argument               | Default  | Description |
    | :--------------------: | :------: | :---- |
    | ``--max_source_size``  |   512    | Optional input sequence length after tokenization. |
    | ``--max_target_size``  |   512    | Optional output sequence length after tokenization. |
    | ``--train_batch_size`` |   64     | Batch size per GPU/CPU for training |
    | ``--eval_batch_size``  |   64     | Batch size per GPU/CPU for evaluation |
    | ``--num_train_epochs`` |  5       | Total number of training epochs to perform |
    | ``--gradient_accumulation_steps`` | 1 | Number of updates steps to accumulate before a backward/update pass. |
    | ``--learning_rate``    |  1e-4    | The initial learning rate for Adam optimizer |
    | ``--weight_decay``     |  0.0     | Weight decay for Adam optimizer |
    | ``--adam_epsilon``     |  1e-8    | Epsilon for Adam optimizer |
    | ``--max_grad_norm``    |  1.0     | Max gradient norm. |
    | ``--seed``             |  42      | Random seed for initialization |
    | ``--beam_size``        |  1       | Beam size for beam search |
  
## Usage Instructions

Follow these instructions:

*Note: All experiments were run on an NVIDIA A6000 GPU, modify ``eval_batch_size`` argument based on your hardware.*


### For Experiments Replication
#### Intrinsic evaluation on executable Python code (RQ1)
   * B1, CodeExecutor + Dynamic Slicing Algorithm
     - Training:
       ```bash

       ```
     - Inference:
       ```bash
       python run.py --data_dir ../data --output_dir ../outputs --encoder unixcoder --decoder unixcoder --do_eval_base --eval_batch_size 512
       ```
       
   * B2, CodeExecutor
     - Training:
       ```bash

       ```
     - Inference:
       ```bash
       python run.py --data_dir ../data --output_dir ../outputs --encoder unixcoder --decoder unixcoder --do_eval --eval_batch_size 512 --load_model_path ../outputs/0.0001/unixcoder_unixcoder/epoch_10/model.ckpt
       ```
       
   * B3, GraphCodeBERT + Pointer-Transformer
     - Training:
       ```bash

       ```
     - Inference:
       ```bash
       python run.py --data_dir ../data --output_dir ../outputs --encoder graphcodebert --decoder transformer --do_eval --eval_batch_size 1024 --use_pointer --load_model_path ../outputs/0.0001/graphcodebert_transformer_pn/epoch_10/model.ckpt
       ```
       
   * B4, GraphCodeBERT + Transformer
     - Training:
       ```bash

       ```
     - Inference:
       ```bash
       python run.py --data_dir ../data --output_dir ../outputs --encoder graphcodebert --decoder transformer --do_eval --eval_batch_size 1024 --load_model_path ../outputs/0.0001/graphcodebert_transformer/epoch_10/model.ckpt
       ```

   * B5, CodeExecutor + Pointer-Transformer
     - Training:
       ```bash

       ```
     - Inference:
       ```bash
       python run.py --data_dir ../data --output_dir ../outputs --encoder unixcoder --decoder transformer --do_eval --eval_batch_size 1024 --use_pointer --load_model_path ../outputs/0.0001/unixcoder_transformer_pn/epoch_10/model.ckpt
       ```

   * B6, CodeExecutor + Pointer-Transformer
     - Training:
       ```bash

       ```
     - Inference:
       ```bash
       python run.py --data_dir ../data --output_dir ../outputs --encoder unixcoder --decoder transformer --do_eval --eval_batch_size 1024 --load_model_path ../outputs/0.0001/unixcoder_transformer/epoch_10/model.ckpt
       ```

#### Intrinsic evaluation on non-executable Python code (RQ2)
   * Table 2 Rows 4-6, B6, CodeExecutor + Transformer
       ```bash
       python run.py --data_dir ../data --output_dir ../outputs --encoder unixcoder --decoder transformer --do_eval_partial --eval_batch_size 512 --load_model_path ../outputs/0.0001/unixcoder_transformer/epoch_10/model.ckpt
       ```

#### Extrinsic evaluation
   * Crash Detection **(RQ3)**
       ```bash
       python run.py --data_dir ../data --output_dir ../outputs --encoder unixcoder --decoder transformer --do_eval_crash --eval_batch_size 1024 --load_model_path ../outputs/0.0001/unixcoder_transformer/epoch_10/model.ckpt 
       ```

#### Qualitative evaluation
   * Statement Types **(RQ4)**
       ```bash
       python run.py --data_dir ../data --output_dir ../outputs --encoder unixcoder --decoder transformer --do_eval_qual --eval_batch_size 1024 --load_model_path ../outputs/0.0001/unixcoder_transformer/epoch_10/model.ckpt
       ```
         
   * Execution Iteration **(RQ5)**
       ```bash
       python run.py --data_dir ../data --output_dir ../outputs --encoder unixcoder --decoder transformer --do_eval_loop --eval_batch_size 1024 --load_model_path ../outputs/0.0001/unixcoder_transformer/epoch_10/model.ckpt
       ```

   * Inter-Procedural Analysis **(RQ6)**
       ```bash
       python run.py --data_dir ../data --output_dir ../outputs --encoder unixcoder --decoder transformer --do_eval_im --eval_batch_size 1024 --load_model_path ../outputs/0.0001/unixcoder_transformer/epoch_10/model.ckpt
       ```
