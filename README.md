## Predictive Program Slicing via Execution Knowledge-Guided Dynamic Dependence Learning

Program slicing, the process of extracting program statements that influence values at a designated location (known as the slicing criterion), is pivotal in both manual and automated debugging. However, such slicing techniques prove ineffective in scenarios where executing specific inputs is prohibitively expensive, or even impossible, as with partial code. In this paper, we introduce ND-Slicer, a predictive slicing methodology that caters to specific executions based on a particular input, overcoming the need for actual execution. We enable such a process by leveraging execution-aware pre-training to learn the dynamic program dependencies, including both dynamic data and control dependencies between variables in the slicing criterion and the remaining program statements. Such knowledge forms the cornerstone for constructing a predictive backward slice. Our empirical evaluation revealed a high accuracy in predicting program slices, achieving an exact-match accuracy of 81.31% and a ROUGE-LCS score of 0.954 on Python programs. As an extrinsic evaluation, we illustrate ND-Slicer usefulness in crash detection, with it locating faults with an accuracy of 63.88%. Furthermore, we include an in-depth qualitative evaluation, assessing ND-Slicer's understanding of branched structures such as if-else blocks and loops, as well as the control flow in inter-procedural calls

### Dataset Links

Here is the link for the dataset used in this paper: [link](https://zenodo.org/record/8062703)

### Model Assets

Here are the links for ND-Slicer with GraphCodeBERT ([link]()) and CodeExecutor ([link]()).

### Getting Started with ND-Slicer

#### Run Instructions

```
$ python run.py --help
usage: run.py [-h] --data_dir DATA_DIR --output_dir OUTPUT_DIR --encoder {unixcoder,graphcodebert} --decoder {unixcoder,graphcodebert,transformer} [--use_pointer] [--do_train] [--do_eval]
              [--do_eval_base] [--do_eval_qual] [--do_eval_loop] [--do_eval_im] [--do_eval_crash] [--do_eval_partial] [--load_model_path LOAD_MODEL_PATH] [--dataset {codenet,bugsinpy}]
              [--config_name CONFIG_NAME] [--max_source_size MAX_SOURCE_SIZE] [--max_target_size MAX_TARGET_SIZE] [--per_gpu_train_batch_size PER_GPU_TRAIN_BATCH_SIZE]
              [--train_batch_size TRAIN_BATCH_SIZE] [--eval_batch_size EVAL_BATCH_SIZE] [--num_train_epochs NUM_TRAIN_EPOCHS] [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]
              [--learning_rate LEARNING_RATE] [--weight_decay WEIGHT_DECAY] [--adam_epsilon ADAM_EPSILON] [--max_grad_norm MAX_GRAD_NORM] [--seed SEED] [--beam_size BEAM_SIZE]

options:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   The input/data caching path
  --output_dir OUTPUT_DIR
                        The output directory where the model predictions and checkpoints will be written.
  --encoder {unixcoder,graphcodebert}
                        Encoder in Seq2Seq framework.
  --decoder {unixcoder,graphcodebert,transformer}
                        Decoder in Seq2Seq framework.
  --use_pointer         Whether to use selective pointer networks.
  --do_train            Whether to run training.
  --do_eval             Whether to run evaluation.
  --do_eval_base        Whether to run baseline evaluation.
  --do_eval_qual        Whether to run qualitative evaluation.
  --do_eval_loop        Whether to run qualitative evaluation on loops.
  --do_eval_im          Whether to run inter-method evaluation.
  --do_eval_crash       Whether to evaluate crash detection.
  --do_eval_partial     Whether to evaluate for partial programs.
  --load_model_path LOAD_MODEL_PATH
                        Path to trained model: Should contain the .bin files
  --dataset {codenet,bugsinpy}
                        Dataset for intrinsic evaluation.
  --config_name CONFIG_NAME
                        Optional pretrained config name or path if not the same as model_name_or_path
  --max_source_size MAX_SOURCE_SIZE
                        Optional input sequence length after tokenization.
  --max_target_size MAX_TARGET_SIZE
                        Optional output sequence length after tokenization.
  --per_gpu_train_batch_size PER_GPU_TRAIN_BATCH_SIZE
                        Batch size per GPU/CPU for training.
  --train_batch_size TRAIN_BATCH_SIZE
                        Batch size per GPU/CPU for evaluation.
  --eval_batch_size EVAL_BATCH_SIZE
                        Batch size per GPU/CPU for evaluation.
  --num_train_epochs NUM_TRAIN_EPOCHS
                        Total number of training epochs to perform.
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        Number of updates steps to accumulate before performing a backward/update pass.
  --learning_rate LEARNING_RATE
                        The initial learning rate for Adam.
  --weight_decay WEIGHT_DECAY
                        Weight deay if we apply some.
  --adam_epsilon ADAM_EPSILON
                        Epsilon for Adam optimizer.
  --max_grad_norm MAX_GRAD_NORM
                        Max gradient norm.
  --seed SEED           random seed for initialization
  --beam_size BEAM_SIZE
                        beam size for beam search
```


#### Sample Commands for Experiment Replication
1. Training
```
python run.py --data_dir <path-to-data> --output_dir <path-to-output> --encoder unixcoder --decoder transformer --do_train --learning_rate 1e-4 --num_train_epochs 10 --train_batch_size 16 --eval_batch_size 16
```
   
2. Inference
```
python run.py --data_dir <path-to-data> --output_dir <path-to-output> --encoder unixcoder --decoder transformer --do_eval --eval_batch_size 16
```

