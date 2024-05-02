'''Main script: Check README.md for example commands to run all experiments in paper.
'''
from __future__ import absolute_import, division, print_function
import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np

from transformers import (
    AdamW, get_linear_schedule_with_warmup, RobertaConfig, RobertaModel, RobertaTokenizer
)

import torch
from torch.nn import TransformerDecoderLayer, TransformerDecoder
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler

from tqdm import tqdm

from dataset import (
    CodeNetTextDataset, PointerNetworkCodeNetTextDataset, PartialCodeNetTextDataset,
    QETextDataset, InterMethodDataset, CrashDetectionDataset
)
from model import Seq2Seq, PointerGeneratedSeq2Seq
from metrics import (
    compute_metrics, compute_metrics_crash, compute_metrics_im, compute_metrics_loop
)


logger = logging.getLogger(__name__)


def set_seed(seed, n_gpu):
    '''Set seed across all platforms to same value.

    Arguments:
        seed (int): Random seed.
        n_gpu (int): CUDA device count.
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def train(args, train_dataset, eval_dataset, model, tokenizer):
    '''Train ND-Slicer model.

    Arguments:
        args (Namespace): Program arguments.
        train_dataset (torch.utils.data.Dataset): Training dataset.
        eval_dataset (torch.utils.data.Dataset): Validation dataset.
        model (model.Seq2Seq or model.PointerGeneratedSeq2Seq): Seq2Seq model.
    '''
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset),
                                  batch_size=args.train_batch_size, drop_last=True)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    max_steps = len(train_dataloader) * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max_steps * 0.1,
                                                num_training_steps=max_steps)
    # Train!
    logger.warning("***** Running training *****")
    logger.warning(f"  Num examples = {len(train_dataset)}")
    logger.warning(
        f"  Total train batch size = {args.train_batch_size * args.gradient_accumulation_steps}")
    logger.warning(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.warning(f"  Total optimization steps = {max_steps}")

    losses, step = [], 0
    model.zero_grad()

    for epoch in range(args.num_train_epochs):
        model.train()
        for batch_id, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            if batch_id == 0:
                print('*** source_ids ***')
                print(batch[0][0])
                print()
                if args.use_pointer:
                    print('*** target_ids ***')
                    print(batch[1][0])
                    print()
                    print('*** gold_ids ***')
                    print(batch[2][0])
                    print()
                else:
                    print('*** gold_ids ***')
                    print(batch[1][0])
                    print()

            # Forward
            batch = tuple(t.to(args.device) for t in batch)
            if not args.use_pointer:
                batch_loss = model(batch[0], batch[1])
            else:
                batch_loss = model(batch[0], batch[3], batch[1])
            # Store loss
            losses.append(batch_loss[0].item())
            # Backward
            batch_loss[0].backward()
            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm)
            # Update parameters
            optimizer.step()
            optimizer.zero_grad()
            # Update the learning rate
            scheduler.step()
            step += 1

            if step % 100 == 0:
                logger.warning(
                    f"Steps: {step} Loss: {round(np.mean(losses), 3)}")
                losses = []

        # Evaluate and save model
        model.eval()
        if args.use_pointer:
            eval_results = evaluate(
                args, eval_dataset, model, tokenizer, use_pointer=True)
        else:
            eval_results = evaluate(args, eval_dataset, model, tokenizer)

        logger.warning(f'Printing evaluation metrics for validation dataset.')
        metrics = compute_metrics(eval_results)
        for k, v in metrics.items():
            if 'ROUGE-LCS' in k:
                print(f'    {k}: {round(v, 3)}')
            else:
                print(f'    {k}: {round(v * 100, 6)}%')
        print()
        if not args.use_pointer:
            eval_results = evaluate(args, eval_dataset, model, tokenizer)
        else:
            eval_results = evaluate_pointer(
                args, eval_dataset, model, tokenizer)

        # Save model checkpoint
        epoch_output_dir = Path(args.output_dir) / f'epoch_{epoch + 1}'
        epoch_output_dir.mkdir(exist_ok=True, parents=True)

        # Saving model predictions
        logger.warning(f'Saving model predictions for validation set.')
        with open(str(epoch_output_dir / "validation-preds.json"), 'w') as f:
            json.dump(eval_results, f, indent=2)

        logger.warning(
            f"Saving optimizer and scheduler states for epoch {epoch + 1} to {epoch_output_dir}")
        torch.save(optimizer.state_dict(), str(
            epoch_output_dir / "optimizer.pt"))
        torch.save(scheduler.state_dict(), str(
            epoch_output_dir / "scheduler.pt"))

        logger.warning(f"Saving model checkpoint to {epoch_output_dir}")
        torch.save(model.state_dict(), str(epoch_output_dir / 'model.ckpt'))
        torch.save(args, str(epoch_output_dir / 'training_args.bin'))


def evaluate(args, eval_dataset, model, tokenizer, use_pointer=False):
    '''Evaluate ND-Slicer performance for intrinsic experiments.

    Arguments:
        args (Namespace): Program arguments.
        eval_dataset (torch.utils.data.Dataset): Validation dataset.
        model (model.Seq2Seq or model.PointerGeneratedSeq2Seq): Trained Seq2Seq model.
        use_pointer (bool): If True, expects ``model`` to be `` an instance of
            ``model.PointerGeneratedSeq2Seq``.
    
    Returns:
        decoded_preds_gold_pairs (dict): Pairs of ground-truth and dynamic slice predictions.
    '''
    eval_dataloader = DataLoader(eval_dataset, sampler=SequentialSampler(eval_dataset),
                                 batch_size=args.eval_batch_size, drop_last=False)
    # Evaluate!
    logger.warning("***** Running evaluation *****")
    logger.warning(f"  Num examples = {len(eval_dataset)}")
    logger.warning(f"  Batch size = {args.eval_batch_size}")

    preds_gold_pairs = []

    model.eval()
    for batch in tqdm(eval_dataloader):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            batch_preds = model(batch[0])
            # Convert ids to labels
            for i, item_preds in enumerate(batch_preds):
                text_topk = []
                for topk_pred in item_preds:
                    t = topk_pred[topk_pred != -999]
                    if use_pointer:
                        t = batch[0][i][t]
                    text_topk.append(t)

                if use_pointer:
                    item_gold = batch[2][i]
                else:
                    item_gold = batch[1][i]
                item_gold = item_gold[item_gold != 1]
                preds_gold_pairs.append(
                    {'preds_topK': text_topk, 'gold': item_gold})

    decoded_preds_gold_pairs = []
    for item in preds_gold_pairs:
        item_preds = tokenizer.decode(item['preds_topK'][0].tolist(),
                                      clean_up_tokenization_spaces=False)
        item_gold = tokenizer.decode(item['gold'].tolist()[1: -1],
                                     clean_up_tokenization_spaces=False)
        decoded_preds_gold_pairs.append(
            {'preds_topK': [item_preds], 'gold': item_gold})

    return decoded_preds_gold_pairs


def evaluate_crash(args, eval_dataset, model, tokenizer):
    '''Evaluate ND-Slicer performance for crash detection.

    Arguments:
        args (Namespace): Program arguments.
        eval_dataset (torch.utils.data.Dataset): Validation dataset.
        model (model.Seq2Seq or model.PointerGeneratedSeq2Seq): Trained Seq2Seq model.
        tokenizer (transformers.RobertaTokenizer): Input tokenizer.
    
    Returns:
        decoded_pairs (dict): Pairs of ground-truth and dynamic slice predictions.
    '''
    eval_dataloader = DataLoader(eval_dataset, sampler=SequentialSampler(eval_dataset),
                                 batch_size=args.eval_batch_size, drop_last=False)
    # Evaluate!
    logger.warning("***** Running evaluation *****")
    logger.warning(f"  Num examples = {len(eval_dataset)}")
    logger.warning(f"  Batch size = {args.eval_batch_size}")

    pairs = []

    model.eval()
    for batch in tqdm(eval_dataloader):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            batch_preds = model(batch[0])
            # Convert ids to labels
            for i, item_preds in enumerate(batch_preds):
                text_topk = []
                for topk_pred in item_preds:
                    t = topk_pred[topk_pred != -999]
                    text_topk.append(t)

                reaching_statements = batch[1][i]
                reaching_statements = reaching_statements[reaching_statements != -999]

                item_gold = batch[2][i]
                item_gold = item_gold[item_gold != -999]
                pairs.append({
                    'preds_topK': text_topk,
                    'reaching_statements': reaching_statements,
                    'gold': item_gold}
                )

    decoded_pairs = []
    for item in pairs:
        item_preds = tokenizer.decode(item['preds_topK'][0].tolist(),
                                      clean_up_tokenization_spaces=False)
        item_gold = item['gold'].tolist()
        reaching_statements = item['reaching_statements'].tolist()
        decoded_pairs.append({
            'preds_topK': [item_preds],
            'reaching_statements': reaching_statements,
            'gold': item_gold}
        )

    return decoded_pairs


def evaluate_im(args, eval_dataset, model, tokenizer):
    '''Evaluate ND-Slicer performance for inter-method analysis.

    Arguments:
        args (Namespace): Program arguments.
        eval_dataset (torch.utils.data.Dataset): Validation dataset.
        model (model.Seq2Seq or model.PointerGeneratedSeq2Seq): Trained Seq2Seq model.
        tokenizer (transformers.RobertaTokenizer): Input tokenizer.
    
    Returns:
        pairs (dict): Pairs of ground-truth and dynamic slice predictions.
    '''
    eval_dataloader = DataLoader(eval_dataset, sampler=SequentialSampler(eval_dataset),
                                 batch_size=args.eval_batch_size, drop_last=False)
    # Evaluate!
    logger.warning("***** Running evaluation *****")
    logger.warning(f"  Num examples = {len(eval_dataset)}")
    logger.warning(f"  Batch size = {args.eval_batch_size}")

    pairs = []

    model.eval()
    for batch in tqdm(eval_dataloader):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            batch_preds = model(batch[0])
            # Convert ids to labels
            for i, item_preds in enumerate(batch_preds):
                text_topk = []
                for topk_pred in item_preds:
                    t = topk_pred[topk_pred != -999]
                    text_topk.append(t)
                decoded = tokenizer.decode(
                    text_topk[0].tolist(), clean_up_tokenization_spaces=False)
                pairs.append({'preds_topK': decoded})
    return pairs


def evaluate_pointer(args, eval_dataset, model, tokenizer):
    '''Evaluate ND-Slicer performance when using ``model.PointerGeneratedSeq2Seq``.

    Arguments:
        args (Namespace): Program arguments.
        eval_dataset (torch.utils.data.Dataset): Validation dataset.
        model (model.Seq2Seq or model.PointerGeneratedSeq2Seq): Trained Seq2Seq model.
        tokenizer (transformers.RobertaTokenizer): Input tokenizer.
    
    Returns:
        decoded_preds_gold_pairs (dict): Pairs of ground-truth and dynamic slice predictions.
    '''
    eval_dataloader = DataLoader(eval_dataset, sampler=SequentialSampler(eval_dataset),
                                 batch_size=args.eval_batch_size, drop_last=False)
    # Evaluate!
    logger.warning("***** Running evaluation *****")
    logger.warning(f"  Num examples = {len(eval_dataset)}")
    logger.warning(f"  Batch size = {args.eval_batch_size}")

    preds_gold_pairs = []

    model.eval()
    for batch in tqdm(eval_dataloader):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            batch_preds = model(batch[0], batch[3])
            # Convert ids to labels
            for i, item_preds in enumerate(batch_preds):
                t = list(item_preds.cpu().numpy())
                if 0 in t:
                    t = t[: t.index(0)]
                text_topk = [tokenizer.decode(
                    t, clean_up_tokenization_spaces=False)]

                item_gold = list(batch[2][i].cpu().numpy())
                item_gold = item_gold[: item_gold.index(
                    tokenizer.sep_token_id) + 1]
                item_gold = tokenizer.decode(
                    item_gold, clean_up_tokenization_spaces=False)
                print(f'{text_topk[0]} *** {item_gold}')
                print('*******')

                preds_gold_pairs.append(
                    {'preds_topK': text_topk, 'gold': item_gold})

    decoded_preds_gold_pairs = []
    for item in preds_gold_pairs:
        item_preds = tokenizer.decode(item['preds_topK'][0].tolist(),
                                      clean_up_tokenization_spaces=False)
        item_gold = tokenizer.decode(item['gold'].tolist()[1: -1],
                                     clean_up_tokenization_spaces=False)
        decoded_preds_gold_pairs.append(
            {'preds_topK': [item_preds], 'gold': item_gold})

    return decoded_preds_gold_pairs


def print_metrics(eval_metrics):
    '''Pretty print all evaluation metrics.

    Arguments:
        eval_metrics (dict): Evaluation results.
    '''
    for k, v in eval_metrics.items():
        if 'ROUGE-LCS' in k:
            print(f'    {k}: {round(v, 3)}')
        else:
            print(f'    {k}: {round(v * 100, 6)}%')
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="Path to nput/data caching")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="Output directory for writing model predictions and checkpoints.")
    parser.add_argument("--encoder", default=None, type=str, required=True,
                        choices=['unixcoder', 'graphcodebert'],
                        help="Encoder in Seq2Seq framework.")
    parser.add_argument("--decoder", default=None, type=str, required=True,
                        choices=['unixcoder', 'graphcodebert', 'transformer'],
                        help="Decoder in Seq2Seq framework.")
    parser.add_argument("--use_pointer", action='store_true',
                        help="Whether to use selective pointer networks.")

    # Run parameters
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run evaluation.")
    parser.add_argument("--do_eval_base", action='store_true',
                        help="Whether to run baseline evaluation.")
    parser.add_argument("--do_eval_qual", action='store_true',
                        help="Whether to run qualitative evaluation.")
    parser.add_argument("--do_eval_loop", action='store_true',
                        help="Whether to run qualitative evaluation on loops.")
    parser.add_argument("--do_eval_im", action='store_true',
                        help="Whether to run inter-method evaluation.")
    parser.add_argument("--do_eval_crash", action='store_true',
                        help="Whether to evaluate crash detection.")
    parser.add_argument("--do_eval_partial", action='store_true',
                        help="Whether to evaluate for partial programs.")
    parser.add_argument("--print_stats", action='store_true',
                        help="Print dataset statistics.")
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files")

    # Other parameters
    parser.add_argument("--dataset", default="codenet", type=str,
                        choices=["codenet", "bugsinpy"], help="Dataset for intrinsic evaluation.")
    parser.add_argument("--config_name", default=None, type=str,
                        help="Pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--max_source_size", default=512, type=int,
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--max_target_size", default=512, type=int,
                        help="Optional output sequence length after tokenization.")
    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--num_train_epochs", default=5, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--beam_size", default=1, type=int,
                        help="beam size for beam search")

    args = parser.parse_args()

    # Setup CUDA, GPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    args.output_dir = Path(args.output_dir) / str(args.learning_rate)
    if not args.use_pointer:
        args.output_dir = args.output_dir / f'{args.encoder}_{args.decoder}'
    else:
        args.output_dir = args.output_dir / f'{args.encoder}_{args.decoder}_pn'

    if not Path(args.output_dir).exists():
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    args.log_file = args.output_dir / 'log.txt'
    if Path(args.log_file).is_file():
        logfile = logging.FileHandler(args.log_file, 'a')
    else:
        logfile = logging.FileHandler(args.log_file, 'w')
    fmt = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s', '%m/%d/%Y %H:%M:%S %p')
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)
    logger.warning(f"Device: {device}, n_gpu: {args.n_gpu}")

    # Set seed
    set_seed(args.seed, args.n_gpu)

    if args.encoder == 'unixcoder':
        tokenizer = RobertaTokenizer.from_pretrained('microsoft/codeexecutor')
    elif args.encoder == 'graphcodebert':
        tokenizer = RobertaTokenizer.from_pretrained(
            'microsoft/graphcodebert-base')
        if args.use_pointer:
            tokenizer = RobertaTokenizer.from_pretrained(
                'microsoft/codeexecutor')
    special_tokens_list = ['<line>', '<state>', '</state>', '<dictsep>', '<output>', '<indent>',
                           '<dedent>', '<mask0>']
    for i in range(200):
        special_tokens_list.append(f"<{i}>")

    if args.use_pointer:
        for i in range(args.max_source_size):
            special_tokens_list.append(f"<unk-{i}>")

    special_tokens_dict = {'additional_special_tokens': special_tokens_list}
    tokenizer.add_special_tokens(special_tokens_dict)

    config = RobertaConfig.from_pretrained('microsoft/codeexecutor')

    if args.decoder == 'unixcoder':
        if args.encoder == 'unixcoder':
            config.is_decoder = True
            encoder = RobertaModel.from_pretrained(
                'microsoft/codeexecutor', config=config)
            encoder.resize_token_embeddings(len(tokenizer))
            decoder = encoder
    elif args.decoder == 'graphcodebert':
        if args.encoder == 'graphcodebert':
            args.max_source_size //= 2
            args.max_target_size //= 2
            config = RobertaConfig.from_pretrained(
                'microsoft/graphcodebert-base')
            config.is_decoder = True
            encoder = RobertaModel.from_pretrained(
                'microsoft/graphcodebert-base', config=config)
            encoder.resize_token_embeddings(len(tokenizer))
            decoder = encoder
    elif args.decoder == 'transformer':
        if args.encoder == 'unixcoder':
            encoder = RobertaModel.from_pretrained(
                'microsoft/codeexecutor', config=config)
            encoder.resize_token_embeddings(len(tokenizer))
        elif args.encoder == 'graphcodebert':
            encoder_config = RobertaConfig.from_pretrained(
                'microsoft/graphcodebert-base')
            encoder = RobertaModel.from_pretrained(
                'microsoft/graphcodebert-base', config=encoder_config)
            encoder.resize_token_embeddings(len(tokenizer))

        decoder_layer = TransformerDecoderLayer(
            d_model=config.hidden_size, nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size, dropout=config.hidden_dropout_prob,
            activation=config.hidden_act, layer_norm_eps=config.layer_norm_eps,
        )
        decoder = TransformerDecoder(decoder_layer, num_layers=6)
    else:
        raise ValueError('Invalid <-Encoder, Decoder-> combination.')

    if not args.use_pointer:
        model = Seq2Seq(
            encoder=encoder, encoder_key=args.encoder, decoder=decoder, decoder_key=args.decoder,
            tokenizer=tokenizer, config=config, beam_size=args.beam_size,
            max_source_length=args.max_source_size, max_target_length=args.max_target_size,
            sos_id=tokenizer.convert_tokens_to_ids(
                ["<mask0>"])[0], eos_id=tokenizer.sep_token_id,
        )
    else:
        model = PointerGeneratedSeq2Seq(
            encoder=encoder, encoder_key=args.encoder, decoder=decoder, tokenizer=tokenizer,
            config=config, beam_size=args.beam_size, max_source_length=args.max_source_size,
            max_target_length=args.max_target_size,
            sos_id=tokenizer.convert_tokens_to_ids(
                ["<mask0>"])[0], eos_id=tokenizer.sep_token_id,
        )

    if args.load_model_path is not None:
        logger.info(f"Reload model from {args.load_model_path}")
        model.load_state_dict(torch.load(args.load_model_path), strict=False)

    model.to(args.device)

    logger.warning(f"Training/evaluation parameters {args}")
    logger.warning(
        f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")

    if args.use_pointer:
        Dataset = PointerNetworkCodeNetTextDataset
    else:
        Dataset = CodeNetTextDataset

    if args.use_pointer:
        Dataset = PointerNetworkCodeNetTextDataset
    else:
        Dataset = CodeNetTextDataset

    if args.do_train:
        train_dataset = Dataset(tokenizer, args, "train", logger)
        logger.warning(f"*** Training Input Example Sample ***")
        for k, v in vars(train_dataset.examples[0]).items():
            print(f'{k}: {v}')
        print()

        eval_dataset = Dataset(tokenizer, args, "val", logger)
        logger.warning(f"*** Validation Input Example Sample ***")
        for k, v in vars(eval_dataset.examples[0]).items():
            print(f'{k}: {v}')
        print()

        # Training
        train(args, train_dataset, eval_dataset, model, tokenizer)

    if args.do_eval:
        eval_dataset = Dataset(tokenizer, args, "test", logger)
        eval_results = evaluate(args, eval_dataset, model,
                                tokenizer, use_pointer=args.use_pointer)
        logger.warning(f" Number of examples = {len(eval_dataset)}")
        logger.warning(f'Printing evaluation metrics for test dataset.')
        metrics = compute_metrics(eval_results)
        print_metrics(metrics)

    if args.do_eval_base:
        eval_dataset = Dataset(tokenizer, args, "test", logger)
        eval_results = evaluate(args, eval_dataset, model,
                                tokenizer, use_pointer=args.use_pointer)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        assert args.encoder == args.decoder == 'unixcoder', "Baseline can only be run for CodeExecutor."

        # Saving model predictions
        logger.warning(f'Saving model predictions with baseline for test set.')
        with open(str(output_dir / "test-base-preds.json"), 'w') as f:
            json.dump(eval_results, f, indent=2)

    if args.do_eval_qual:
        eval_dataset_if = QETextDataset(
            tokenizer, args, 'test', logger, 'if_statement', partial=args.do_eval_partial)
        logger.warning(
            f" Number of examples with if-condition = {len(eval_dataset_if)}")

        eval_dataset_for = QETextDataset(
            tokenizer, args, 'test', logger, 'for_statement', partial=args.do_eval_partial)
        logger.warning(
            f" Number of examples with for-loop = {len(eval_dataset_for)}")

        eval_dataset_while = QETextDataset(
            tokenizer, args, 'test', logger, 'while_statement', partial=args.do_eval_partial)
        logger.warning(
            f" Number of examples with while-loop = {len(eval_dataset_while)}")

        if_ids = set([ex.id for ex in eval_dataset_if.examples])
        for_ids = set([ex.id for ex in eval_dataset_for.examples])
        while_ids = set([ex.id for ex in eval_dataset_while.examples])
        loop_ids = set(for_ids).union(while_ids)

        only_if_ids = set(
            [ex.id for ex in eval_dataset_if.examples if ex.id not in loop_ids])
        only_for_ids = set([eid for eid in for_ids if eid not in if_ids])

        all_if_examples = eval_dataset_if.examples
        condition_examples = [
            ex for ex in eval_dataset_if.examples if ex.id in only_if_ids]
        condition_loop_examples = [
            ex for ex in eval_dataset_if.examples if ex.id in loop_ids]

        all_for_examples = eval_dataset_for.examples
        only_for_examples = [
            ex for ex in eval_dataset_for.examples if ex.id in only_for_ids]

        all_while_examples = eval_dataset_while.examples

        eval_dataset_if.examples = all_if_examples
        eval_results_if = evaluate(
            args, eval_dataset_if, model, tokenizer, use_pointer=args.use_pointer)
        logger.warning(
            f'Printing evaluation metrics for test dataset with if-condition.')
        metrics = compute_metrics(eval_results_if)
        print_metrics(metrics)

        eval_dataset_if.examples = condition_examples
        eval_results_if = evaluate(
            args, eval_dataset_if, model, tokenizer, use_pointer=args.use_pointer)
        logger.warning(
            f" Number of examples with only if-condition = {len(eval_dataset_if)}")
        logger.warning(
            f'Printing evaluation metrics for test dataset with only if-condition.')
        metrics = compute_metrics(eval_results_if)
        print_metrics(metrics)

        eval_dataset_if.examples = condition_loop_examples
        eval_results_if = evaluate(
            args, eval_dataset_if, model, tokenizer, use_pointer=args.use_pointer)
        logger.warning(
            f" Number of examples with if-condition in loops = {len(eval_dataset_if)}")
        logger.warning(
            f'Printing evaluation metrics for test dataset with if-condition in loops.')
        metrics = compute_metrics(eval_results_if)
        print_metrics(metrics)

        eval_dataset_for.examples = all_for_examples
        eval_results_for = evaluate(
            args, eval_dataset_for, model, tokenizer, use_pointer=args.use_pointer)
        logger.warning(
            f'Printing evaluation metrics for test dataset with for loop.')
        metrics = compute_metrics(eval_results_for)
        print_metrics(metrics)

        eval_dataset_for.examples = only_for_examples
        eval_results_for = evaluate(
            args, eval_dataset_for, model, tokenizer, use_pointer=args.use_pointer)
        logger.warning(
            f'Printing evaluation metrics for test dataset with for loop without if-conditions.')
        metrics = compute_metrics(eval_results_for)
        print_metrics(metrics)

        eval_dataset_while.examples = all_while_examples
        eval_results_while = evaluate(
            args, eval_dataset_while, model, tokenizer, use_pointer=args.use_pointer)
        logger.warning(
            f'Printing evaluation metrics for test dataset with while loop.')
        metrics = compute_metrics(eval_results_while)
        print_metrics(metrics)

        if not args.do_eval_partial:
            eval_dataset = CodeNetTextDataset(tokenizer, args, "test", logger)
        else:
            eval_dataset = PartialCodeNetTextDataset(
                tokenizer, args, 'test', logger)

        branchless_examples = [
            ex for ex in eval_dataset.examples if ex.id not in loop_ids]
        eval_dataset.examples = branchless_examples
        eval_results = evaluate(args, eval_dataset, model,
                                tokenizer, use_pointer=args.use_pointer)
        logger.warning(
            f" Number of examples with no branches = {len(eval_dataset)}")
        logger.warning(f'Printing evaluation metrics for test dataset.')
        metrics = compute_metrics(eval_results)
        print_metrics(metrics)

    if args.do_eval_loop:
        eval_dataset_if = QETextDataset(
            tokenizer, args, 'test', logger, 'if_statement', partial=args.do_eval_partial)
        all_if_examples = eval_dataset_if.examples

        eval_dataset_for = QETextDataset(
            tokenizer, args, 'test', logger, 'for_statement', partial=args.do_eval_partial)
        all_for_examples = eval_dataset_for.examples

        all_if_ids = set([ex.id for ex in eval_dataset_if.examples])
        all_for_ids = set([ex.id for ex in eval_dataset_for.examples])

        occurrences_examples = {}
        for ex in all_for_examples:
            if 1 <= int(ex.occurrence) <= 5:
                key = str(ex.occurrence)
            elif 6 <= int(ex.occurrence) <= 10:
                key = "6 -- 10"
            elif int(ex.occurrence) > 10:
                key = "> 10"

            if key in occurrences_examples:
                occurrences_examples[key] += [ex]
            else:
                occurrences_examples[key] = [ex]

        total = sum([len(_examples)
                    for _examples in occurrences_examples.values()])
        for occurrence_key, _examples in occurrences_examples.items():
            eval_dataset_for.examples = _examples
            eval_results_for = evaluate(
                args, eval_dataset_for, model, tokenizer, use_pointer=args.use_pointer)
            metrics, _ = compute_metrics_loop(eval_results_for)
            logger.warning(
                f" Number of examples with for-loop for occurrence {occurrence_key} = {len(eval_dataset_for)}")
            logger.warning(
                f" % of examples with for-loop for occurrence {occurrence_key} = {round(len(_examples) * 100/ total, 3)}")
            logger.warning(
                f'Printing evaluation metrics for test dataset: for-overall')
            print_metrics(metrics)

            for_without_if_ids = set(
                [ex.id for ex in _examples]).difference(all_if_ids)
            eval_dataset_for.examples = [
                ex for ex in _examples if ex.id in for_without_if_ids]
            eval_results_for = evaluate(
                args, eval_dataset_for, model, tokenizer, use_pointer=args.use_pointer)
            metrics, _ = compute_metrics_loop(eval_results_for)
            logger.warning(
                f" Number of examples with for-loop for occurrence {occurrence_key} without if-statement = {len(eval_dataset_for)}")
            logger.warning(
                f" % of examples with for-loop for occurrence {occurrence_key} = {round(len(_examples) * 100/ total, 3)}")
            logger.warning(
                f'Printing evaluation metrics for test dataset: for-without-if')
            print_metrics(metrics)

            for_with_if_ids = set(
                [ex.id for ex in _examples]).intersection(all_if_ids)
            eval_dataset_for.examples = [
                ex for ex in _examples if ex.id in for_with_if_ids]
            eval_results_for = evaluate(
                args, eval_dataset_for, model, tokenizer, use_pointer=args.use_pointer)
            metrics, _ = compute_metrics_loop(eval_results_for)
            logger.warning(
                f" Number of examples with for-loop for occurrence {occurrence_key} with if-statement = {len(eval_dataset_for)}")
            logger.warning(
                f" % of examples with for-loop for occurrence {occurrence_key} = {round(len(_examples) * 100/ total, 3)}")
            logger.warning(
                f'Printing evaluation metrics for test dataset: for-with-if.')
            print_metrics(metrics)

    if args.do_eval_im:
        eval_dataset = InterMethodDataset(tokenizer, args, 'test', logger)
        eval_results = evaluate_im(args, eval_dataset, model, tokenizer)
        logger.warning(f" Number of examples = {len(eval_dataset)}")
        logger.warning(f'Printing evaluation metrics for test dataset.')
        metrics, cls = compute_metrics_im(eval_results, eval_dataset.examples)
        print_metrics(metrics)

        assert len(eval_dataset_for.examples) == len(cls)

        stratified_examples = []
        for eid, ex in enumerate(eval_dataset.examples):
            if cls[eid]['label'] == 'correct':
                item_output = cls[eid]
                item_output['id'] = ex.id
                stratified_examples.append(item_output)
        with open('stratified-im.json', 'w') as f:
            json.dump(stratified_examples, f, indent=2)

    if args.do_eval_crash:
        eval_dataset = CrashDetectionDataset(tokenizer, args, 'test', logger)
        eval_results = evaluate_crash(args, eval_dataset, model, tokenizer)
        logger.warning(f" Number of examples = {len(eval_dataset)}")

        logger.warning(f'Printing evaluation metrics for test dataset.')
        metrics = compute_metrics_crash(eval_results)
        print_metrics(metrics)

    if args.do_eval_partial:
        eval_dataset_partial = PartialCodeNetTextDataset(
            tokenizer, args, 'test', logger)
        all_examples = eval_dataset_partial.examples
        for k, v in vars(eval_dataset_partial.examples[0]).items():
            print(f'{k}: {v}')
        print()

        eval_results_partial = evaluate(
            args, eval_dataset_partial, model, tokenizer)
        logger.warning(f" Number of examples = {len(eval_dataset_partial)}")
        logger.warning(f'Printing evaluation metrics for test dataset.')
        metrics_partial = compute_metrics(eval_results_partial)
        print_metrics(metrics_partial)

        partial_new_ids = set([ex.id for ex in eval_dataset_partial.examples])
        partial_ids = set(
            [ex.original_id for ex in eval_dataset_partial.examples])
        eval_dataset_complete = CodeNetTextDataset(
            tokenizer, args, 'test', logger)
        eval_dataset_complete.examples = [
            ex for ex in eval_dataset_complete.examples if ex.id in partial_ids]
        eval_results_complete = evaluate(
            args, eval_dataset_complete, model, tokenizer)
        logger.warning(f" Number of examples = {len(eval_dataset_complete)}")
        logger.warning(f'Printing evaluation metrics for test dataset.')
        metrics_complete = compute_metrics(eval_results_complete)
        print_metrics(metrics_complete)

        eval_dataset_if = QETextDataset(
            tokenizer, args, 'test', logger, 'if_statement', partial=True)
        logger.warning(
            f" Number of examples with if-condition = {len(eval_dataset_if)}")

        eval_dataset_for = QETextDataset(
            tokenizer, args, 'test', logger, 'for_statement', partial=True)
        logger.warning(
            f" Number of examples with for-loop = {len(eval_dataset_for)}")

        eval_dataset_while = QETextDataset(
            tokenizer, args, 'test', logger, 'while_statement', partial=True)
        logger.warning(
            f" Number of examples with while-loop = {len(eval_dataset_while)}")

        if_ids = set([ex.id for ex in eval_dataset_if.examples])
        for_ids = set([ex.id for ex in eval_dataset_for.examples])
        while_ids = set([ex.id for ex in eval_dataset_while.examples])
        loop_ids = set(for_ids).union(while_ids)
        branched_ids = if_ids.union(loop_ids)
        branchless_ids = partial_new_ids.difference(branched_ids)

        eval_dataset_partial.examples = [
            ex for ex in all_examples if ex.id in branchless_ids]
        eval_results_partial_branchless = evaluate(
            args, eval_dataset_partial, model, tokenizer)
        logger.warning(f" Number of examples = {len(eval_dataset_partial)}")
        logger.warning(
            f'Printing evaluation metrics for test dataset with NO branches.')
        metrics_partial_branchless = compute_metrics(
            eval_results_partial_branchless)
        print_metrics(metrics_partial_branchless)

        eval_dataset_partial.examples = [
            ex for ex in all_examples if ex.id in branched_ids]
        eval_results_partial_branched = evaluate(
            args, eval_dataset_partial, model, tokenizer)
        logger.warning(f" Number of examples = {len(eval_dataset_partial)}")
        logger.warning(
            f'Printing evaluation metrics for test dataset with branches.')
        metrics_partial_branched = compute_metrics(
            eval_results_partial_branched)
        print_metrics(metrics_partial_branched)
