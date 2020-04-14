# -*- coding: utf-8 -*-
import argparse
import sys
from revolut.log import _logger
import logging
from revolut.utils import ETL

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm.gui import trange
from transformers import DistilBertConfig, DistilBertTokenizer, InputFeatures, AdamW, get_linear_schedule_with_warmup, \
    DistilBertPreTrainedModel, DistilBertModel
from revolut.cfg import *
from revolut import env
import torch
from torch import nn
from tqdm import tqdm
import random
import numpy as np

__author__ = "yj"
__copyright__ = "yj"
__license__ = "mit"


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


class DistilBertForSequenceClassification(DistilBertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]
    """

    def __init__(self, config):
        super(DistilBertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, head_mask=None, inputs_embeds=None, labels=None):
        distilbert_output = self.distilbert(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            head_mask=head_mask,
                                            inputs_embeds=inputs_embeds)
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, dim)

        outputs = (logits,) + distilbert_output[1:]
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


def training_model(args):
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    config = DistilBertConfig.from_pretrained(TRANSFORMER_MODEL, num_labels=97 + 1)
    tokenizer = DistilBertTokenizer.from_pretrained(TRANSFORMER_MODEL)
    model = DistilBertForSequenceClassification.from_pretrained(TRANSFORMER_MODEL, config=config)
    model.to(args.device)
    etl = ETL(env.DB_FILE, env.SCHEMA_FILE)
    complaints_users = etl.load_query(SQL_QUERY_STRING)
    features = convert_examples_to_features(
        complaints_users[[COMPLAINT_TEXT, LABEL]].to_dict(orient='records'),
        max_length=128,
        tokenizer=tokenizer)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    train_dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=32)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    t_total = len(train_dataloader) // args.num_train_epochs
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(epochs_trained, int(args.num_train_epochs), desc="Epoch")
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[3]}
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            loss.backward()

            tr_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1

            logs = {}
            if args.evaluate_during_training:
                results = evaluate(args, model, tokenizer)
                for key, value in results.items():
                    eval_key = 'eval_{}'.format(key)
                    logs[eval_key] = value

            loss_scalar = (tr_loss - logging_loss) / args.logging_steps
            learning_rate_scalar = scheduler.get_lr()[0]
            logs['learning_rate'] = learning_rate_scalar
            logs['loss'] = loss_scalar
            logging_loss = tr_loss

    model.eval()

    # Creating the trace
    dummy_all_input_ids = torch.tensor([f.input_ids for f in features[0:1]], dtype=torch.long).to(args.device)
    dummy_all_attention_mask = torch.tensor([f.attention_mask for f in features[0:1]], dtype=torch.long).to(args.device)

    traced_model = torch.jit.trace(model, [dummy_all_input_ids, dummy_all_attention_mask])
    torch.jit.save(traced_model, "traced_bert.pt")
    tokenizer.save_pretrained('tokenizer')


def convert_examples_to_features(examples, tokenizer,
                                 max_length=512,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    features = []
    for (ex_index, example) in tqdm(enumerate(examples)):
        inputs = tokenizer.encode_plus(
            example[COMPLAINT_TEXT],
            add_special_tokens=True,
            max_length=max_length,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask),
                                                                                            max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids),
                                                                                            max_length)

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          label=example[LABEL]))
    return features


def get_args_parser():
    """
    Define the argument parser

    Returns:
        argparse.Namespace: command line parameters namespace
    """
    parser = argparse.ArgumentParser(description='Start a Model Training or Prediction')
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO)
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG)

    subparser = parser.add_subparsers(dest='command')

    train = subparser.add_parser('train', help='model training')
    train.add_argument("--learning_rate", default=5e-5, type=float,
                       help="The initial learning rate for Adam.")
    train.add_argument("--weight_decay", default=0.0, type=float,
                       help="Weight decay if we apply some.")
    train.add_argument("--adam_epsilon", default=1e-8, type=float,
                       help="Epsilon for Adam optimizer.")
    train.add_argument("--max_grad_norm", default=1.0, type=float,
                       help="Max gradient norm.")
    train.add_argument("--num_train_epochs", default=1.0, type=float,
                       help="Total number of training epochs to perform.")
    train.add_argument("--max_steps", default=-1, type=int,
                       help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    train.add_argument("--warmup_steps", default=0, type=int,
                       help="Linear warmup over warmup_steps.")
    train.add_argument('--seed', type=int, default=env.SEED,
                       help="random seed for initialization")
    train.add_argument("--evaluate_during_training", action='store_true',
                       help="Rul evaluation during training at each logging step.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")

    predict = subparser.add_parser('predict', help='predict')
    predict.add_argument("-i", "--input", help='string to predict', required=True)
    return parser


def get_run_args():
    """
    Print running arguments

    Returns:
         argparse.Namespace: command line parameters namespace
    """
    args, unknowns = get_args_parser().parse_known_args()
    param_str = '\n'.join(['%20s = %s' % (k, v) for k, v in sorted(vars(args).items())])
    print('usage: %s\n%20s   %s\n%s\n%s\n' % (' '.join(sys.argv), 'ARG', 'VALUE', '_' * 50, param_str))
    return args


def setup_logging(loglevel):
    """
    Setup basic logging

    Args:
        loglevel (int): minimum loglevel for emitting messages

    """

    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format=logformat, datefmt="%Y-%m-%d %H:%M:%S")


def main():
    """
    Main entry Point for training and prediction

    Example:
        - python main.py train
        - python main.py predict --input "My input"
    """

    args = get_run_args()
    setup_logging(args.loglevel)
    if args.command == 'train':
        _logger.debug("Starting Training Process")
        training_model(args)
    elif args.command == 'predict':
        print(args.input)
    else:
        raise ValueError('Unknown command : %s ' % args.command)


if __name__ == "__main__":
    main()
