# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
import yaml
import os
import numpy as np
import argparse
import json
import sys
from tqdm.auto import tqdm
from data_utils import load_data
from data_utils.task_def import TaskType, DataFormat
from data_utils.log_wrapper import create_logger
from experiments.exp_def import TaskDefs, EncoderModelType
from data_utils.my_statics import DUMPY_STRING_FOR_EMPTY_ANS
from transformers import AutoTokenizer

DEBUG_MODE = False
MAX_SEQ_LEN = 384
DOC_STRIDE = 180
MAX_QUERY_LEN = 64
TGT_MAX_SEQ_LEN = 4

logger = create_logger(
    __name__,
    to_disk=True,
    log_file='mt_dnn_clues_gen_data_proc_{}.log'.format(MAX_SEQ_LEN))

def load_clues_jsonl(path, data_type, is_training=True):
    examples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if len(line) < 1: return examples
            obj = json.loads(line)
            is_impossible = obj.get("is_impossible", None)
            answers = obj.get("answer", [])
            context = obj.get('context', None)
            question = obj.get('question', None)
            if data_type == DataFormat.CLUE_CLASSIFICATION:
                new_answers = []
                for answer in answers:
                    new_answer = {"text": [answer], "answer_start": [question.find(answer)]}
                    new_answers.append(new_answer)
                answers = new_answer
            elif data_type == DataFormat.CLUE_SPAN:
                if len(answers) > 0:
                    answer_text = [answer['text'].strip() for answer in answers]
                    answer_start = [answer['answer_start'] for answer in answers]
                else:
                    answer_text = []
                    answer_start = []
                answers = {'text': answer_text, 'answer_start': answer_start}
            else:
                if len(answers) > 0:
                    new_answers = []
                    for answer in answers:
                        new_answer = {"text": answer, "answer_start": context.find(answer)}
                        new_answers.append(new_answer)
                    answers = sorted(new_answers, key = lambda x : x['answer_start'])
                    answers = {'text': [", ".join([ans['text'] for ans in answers])]}
                else:
                    answers = {"text": [DUMPY_STRING_FOR_EMPTY_ANS]} if is_training else []
                
            obj['answer'] = answers
            obj['is_impossible'] = is_impossible
            examples.append(obj)
        return examples

def prepare_train_feature(tokenizer, samples, output_path, data_type=DataFormat.CLUE_GEN, max_seq_length=384, doc_stride=128, tgt_max_seq_length=64, pad_on_right=True, pad_to_max_length=True, label_mapper=None):
    with open(output_path, 'w', encoding='utf-8') as writer:
        for sample in samples:
            context = sample['context']
            question = sample['question']
            # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
            # in one example possible giving several features when a context is long, each of those features having a
            # context that overlaps a bit the context of the previous feature.
            tokenized_examples = tokenizer(
                question if pad_on_right else context,
                context if pad_on_right else question,
                truncation="only_second" if pad_on_right else "only_first",
                max_length=max_seq_length,
                stride=doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length" if pad_to_max_length else False,
            )

            # The offset mappings will give us a map from token to character position in the original context. This will
            # help us compute the start_positions and end_positions.
            offset_mapping = tokenized_examples.pop("offset_mapping")

            # Let's label those examples!
            tokenized_examples["id"] = []
            tokenized_examples["label"] = []
            for i, _ in enumerate(offset_mapping):
                # We will label impossible answers with the index of the CLS token.
                if tokenizer.cls_token_id:
                    cls_token_id = tokenizer.cls_token_id
                else:
                    cls_token_id = tokenizer.eos_token_id

                # One example can give several spans, this is the index of the example containing this span of text.
                answers = sample['answer']
                tokenized_examples["id"].append(sample['id'])
                # If no answers are given, set the cls_index as answer.
                if len(answers) == 0 or len(answers["text"]) == 0:
                    # tokenized_examples["label"].append(cls_index)
                    label = [cls_token_id]
                else:
                    tokenized_tgt_examples = tokenizer(
                        answers["text"][0],
                        max_length=tgt_max_seq_length,
                        truncation="only_first",
                        return_overflowing_tokens=False,
                        return_offsets_mapping=False,
                        padding="max_length" if pad_to_max_length else False,
                    )
                    label = [tokenizer.pad_token_id] + tokenized_tgt_examples["input_ids"]

                while len(label) < tgt_max_seq_length + 1:
                    label.append(tokenizer.pad_token_id)    
                tokenized_examples["label"].append(label)
                      
            for i in range(0, len(tokenized_examples['input_ids'])):
                sample = {'uid': tokenized_examples['id'][i],
                'token_id' : tokenized_examples['input_ids'][i],
                'mask': tokenized_examples['attention_mask'][i],
                'type_id': tokenized_examples['token_type_ids'][i] if "token_type_ids" in tokenized_examples else len(tokenized_examples['input_ids'][i]) * [0],
                'label': tokenized_examples['label'][i]
                }
                writer.write('{}\n'.format(json.dumps(sample)))

# Validation preprocessing
def prepare_validation_features(tokenizer, samples, output_path, data_type=DataFormat.CLUE_CLASSIFICATION, max_seq_length=384, doc_stride=128, tgt_max_seq_length=64, pad_on_right=True, pad_to_max_length=True, label_mapper=None):
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    answer_in_query = True if data_type == DataFormat.CLUE_CLASSIFICATION else False

    with open(output_path, 'w', encoding='utf-8') as writer:
        for sample in samples:
            context = sample['context']
            question = sample['question']
            answer = sample.get("answer")
            # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
            # in one example possible giving several features when a context is long, each of those features having a
            # context that overlaps a bit the context of the previous feature.
            tokenized_examples = tokenizer(
                question if pad_on_right else context,
                context if pad_on_right else question,
                truncation="only_second" if pad_on_right else "only_first",
                max_length=max_seq_length,
                stride=doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length" if pad_to_max_length else False,
            )

            # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
            # corresponding example_id and we will store the offset mappings.
            tokenized_examples["id"] = []
            tokenized_examples["label"] = []

            for i in range(len(tokenized_examples["input_ids"])):
                if tokenizer.cls_token_id:
                    cls_token_id = tokenizer.cls_token_id
                else:
                    cls_token_id = tokenizer.eos_token_id

                # One example can give several spans, this is the index of the example containing this span of text.
                answers = sample['answer']
                tokenized_examples["id"].append(sample['id'])
                if 'answer' in sample:
                    # If no answers are given, set the cls_index as answer.
                    if len(answers) == 0 or len(answers["text"]) == 0:
                        label = [cls_token_id]
                    else:
                        tokenized_tgt_examples = tokenizer(
                            answers["text"][0],
                            max_length=tgt_max_seq_length,
                            return_overflowing_tokens=False,
                            truncation="only_first",
                            return_offsets_mapping=False,
                            padding="max_length" if pad_to_max_length else False,
                        )
                        label = [tokenizer.pad_token_id] + tokenized_tgt_examples["input_ids"]
                else:
                    label = [cls_token_id] 
                while len(label) < tgt_max_seq_length + 1:
                    label.append(tokenizer.pad_token_id)    
                tokenized_examples["label"].append(label)

            if sample['is_impossible'] is not None:
                answers['is_impossible'] = sample['is_impossible']

            for i in range(0, len(tokenized_examples['input_ids'])):
                sample = {'uid': tokenized_examples['id'][i],
                'token_id' : tokenized_examples['input_ids'][i],
                'mask': tokenized_examples['attention_mask'][i],
                'type_id': tokenized_examples['token_type_ids'][i] if "token_type_ids" in tokenized_examples else len(tokenized_examples['input_ids'][i]) * [0],
                'answer': answers,
                'label': tokenized_examples['label'][i]}
                writer.write('{}\n'.format(json.dumps(sample)))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Preprocessing MRC dataset.')
    parser.add_argument('--model', type=str, default='bert-base-uncased',
                        help='support all BERT, XLNET and ROBERTA family supported by HuggingFace Transformers')
    parser.add_argument('--root_dir', type=str, default='data/canonical_data')
    parser.add_argument('--cache_dir', type=str, default=".cache")
    parser.add_argument('--model_revision', type=str, default=None)
    parser.add_argument('--task_def', type=str, default="experiments/clue/clue_task_def.yml")
    parser.add_argument('--max_seq_length', type=int, default=MAX_SEQ_LEN)
    parser.add_argument('--doc_stride', type=int, default=DOC_STRIDE)
    parser.add_argument('--tgt_max_seq_length', type=int, default=TGT_MAX_SEQ_LEN)

    args = parser.parse_args()
    return args



def main(args):
    # hyper param
    root = args.root_dir
    assert os.path.exists(root)
    suffix = args.model.split('/')[-1]
    literal_model_type = suffix.split('-')[0].upper()
    if 'UNILM' in literal_model_type:
        literal_model_type = 'UNILM'

    literal_model_type = literal_model_type.lower()
    mt_dnn_suffix = literal_model_type
    if 'base' in args.model:
        mt_dnn_suffix += "_base"
    elif 'large' in args.model:
        mt_dnn_suffix += "_large"

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model,
        cache_dir=args.cache_dir,
        use_fast=True,
        revision=args.model_revision
    )
    # Padding side determines if we do (question|context) or (context|question).
    # pad_on_right = tokenizer.padding_side == "right"
    pad_on_right = False

    if 'uncased' in args.model:
        mt_dnn_suffix = '{}_uncased'.format(mt_dnn_suffix)
    else:
        mt_dnn_suffix = '{}_cased'.format(mt_dnn_suffix)
    # gen
    mt_dnn_suffix = '{}_gen'.format(mt_dnn_suffix)
    mt_dnn_root = os.path.join(root, mt_dnn_suffix)
    if not os.path.isdir(mt_dnn_root):
        os.mkdir(mt_dnn_root)

    task_defs = TaskDefs(args.task_def)

    for task in task_defs.get_task_names():
        task_def = task_defs.get_task_def(task)
        logger.info("Task %s" % task)
        for split_name in task_def.split_names:
            file_path = os.path.join(root, "%s_%s.json" % (task, split_name))
            print(file_path)

            if not os.path.exists(file_path):
                logger.warning("File %s doesnot exit")
                sys.exit(1)
            logger.warning("processing %s" % file_path)
            is_training = True
            if not "train" in split_name:
                is_training = False
            rows = load_clues_jsonl(file_path, task_def.data_type)
            dump_path = os.path.join(mt_dnn_root, "%s_%s.json" % (task, split_name))
            logger.info(dump_path)
            if is_training:
                prepare_train_feature(tokenizer, rows, dump_path, data_type=task_def.data_type, pad_on_right=pad_on_right, label_mapper=task_def.label_vocab, max_seq_length=args.max_seq_length, doc_stride=args.doc_stride, tgt_max_seq_length=args.tgt_max_seq_length)
            else:
                prepare_validation_features(tokenizer, rows, dump_path, data_type=task_def.data_type, pad_on_right=pad_on_right, label_mapper=task_def.label_vocab, max_seq_length=args.max_seq_length, doc_stride=args.doc_stride, tgt_max_seq_length=args.tgt_max_seq_length)



if __name__ == '__main__':
    args = parse_args()
    main(args)
