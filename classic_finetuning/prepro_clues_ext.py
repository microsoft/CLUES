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
from transformers import AutoTokenizer
# from .clues_utils import load_clues_jsonl, flat_clues_cls, flat_clues_qa, flat_clues_ner

DEBUG_MODE = False
MAX_SEQ_LEN = 384
DOC_STRIDE = 180
MAX_QUERY_LEN = 64

logger = create_logger(
    __name__,
    to_disk=True,
    log_file='mt_dnn_clues_data_proc_{}.log'.format(MAX_SEQ_LEN))

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
                        if len(answer) < 1: continue
                        new_answer = {"text": answer, "answer_start": context.find(answer)}
                        new_answers.append(new_answer)
                    answers = sorted(new_answers, key = lambda x : x['answer_start'])
                    answers = {'text': [ans['text'] for ans in answers],
                        'answer_start': [ans['answer_start'] for ans in answers]
                    }
                
            obj['answer'] = answers
            obj['is_impossible'] = is_impossible
            examples.append(obj)
        return examples


def search_index(input_ids, sequence_ids, offsets, cls_index, start_char, end_char, answer_in_query=False, pad_on_right=False):
    start_position, end_position = cls_index, cls_index
    # Start token index of the current span in the text.
    token_start_index = 0
    if answer_in_query:
        while sequence_ids[token_start_index] != (0 if pad_on_right else 1):
            token_start_index += 1                        
    else:
        while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
            token_start_index += 1
    # End token index of the current span in the text.
    token_end_index = len(input_ids) - 1
    if answer_in_query:
        while sequence_ids[token_end_index] != (0 if pad_on_right else 1):
            token_end_index -= 1
    else:
        while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
            token_end_index -= 1

    # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
    if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
        start_position = cls_index
        end_position = cls_index
    else:
        # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
        # Note: we could go after the last offset if the answer is the last word (edge case).
        while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
            token_start_index += 1
        start_position = token_start_index - 1
        while offsets[token_end_index][1] >= end_char:
            token_end_index -= 1
        end_position = token_end_index + 1
    return start_position, end_position      


def fill_labels(labels, start, end, label_mapper):
    assert start < len(labels) and end < len(labels)
    # B
    assert 'B-ENT' in label_mapper and 'I-ENT' in label_mapper
    labels[start] = 'B-ENT'
    for i in range(start + 1, end + 1):
        labels[i] = 'I-ENT'
    return labels


def prepare_train_feature(tokenizer, samples, output_path, data_type=DataFormat.CLUE_CLASSIFICATION, max_seq_length=384, doc_stride=128, pad_on_right=True, pad_to_max_length=True, label_mapper=None):
    if not tokenizer.cls_token:
        # cls_tok_id = tokenizer.eos_token_id
        cls_tok_id = tokenizer.pad_token_id
        prefix_pad = True
    else:
        cls_tok_id = tokenizer.cls_token_id
        prefix_pad = False

    if not tokenizer.sep_token:
        sep_tok_id = tokenizer.eos_token_id
        sep_tok = tokenizer.eos_token
    else:
        sep_tok_id = tokenizer.sep_token_id
        sep_tok = tokenizer.sep_token

    with open(output_path, 'w', encoding='utf-8') as writer:
        for sample in samples:
            context = sample['context']
            if isinstance(context, list):
                context = " {} ".format(sep_tok).join(context)
            question = sample['question']

            if pad_on_right and prefix_pad:
                question = tokenizer.pad_token + question

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
                verbose=False,
            )
           
            # Since one example might give us several features if it has a long context, we need a map from a feature to
            # its corresponding example. This key gives us just that.
            sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
            # The offset mappings will give us a map from token to character position in the original context. This will
            # help us compute the start_positions and end_positions.
            offset_mapping = tokenized_examples.pop("offset_mapping")

            answer_in_query = True if data_type == DataFormat.CLUE_CLASSIFICATION else False

            # Let's label those examples!
            tokenized_examples["start_positions"] = []
            tokenized_examples["end_positions"] = []
            tokenized_examples["id"] = []
            tokenized_examples["label"] = []
            for i, offsets in enumerate(offset_mapping):
                # We will label impossible answers with the index of the CLS token.
                input_ids = tokenized_examples["input_ids"][i]

                # Grab the sequence corresponding to that example (to know what is the context and what is the question).
                sequence_ids = tokenized_examples.sequence_ids(i)

                cls_index = input_ids.index(cls_tok_id)

                # One example can give several spans, this is the index of the example containing this span of text.
                # sample_index = sample_mapping[i]
                answers = sample['answer']
                tokenized_examples["id"].append(sample['id'])
                if data_type != DataFormat.CLUE_SEQ:
                    label = None
                    if sample.get('is_impossible', None):
                        label = 1 if sample['is_impossible'] else 0
                    tokenized_examples["label"].append(label)
                # If no answers are given, set the cls_index as answer.
                if len(answers) == 0 or len(answers["answer_start"]) == 0:
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                    if data_type == DataFormat.CLUE_SEQ:
                        labels = ['O'] * len(input_ids)
                        labels = label_mapper.toidx(labels)
                    else:
                        labels = 1
                    tokenized_examples["label"].append(labels)
                else:
                    # Start/end character index of the answer in the text.
                    if data_type == DataFormat.CLUE_SEQ:
                        start_positions, end_positions = [], []
                        labels = ['O'] * len(input_ids)
                        for aidx, start_char in enumerate(answers["answer_start"]):
                            end_char = start_char + len(answers["text"][aidx])
                            start_position, end_position = search_index(input_ids, sequence_ids, offsets, cls_index, start_char, end_char, answer_in_query=answer_in_query, pad_on_right=pad_on_right)
                            start_positions.append(start_position)
                            end_positions.append(end_position)
                            labels = fill_labels(labels, start_position, end_position, label_mapper)
                        tokenized_examples["start_positions"].append(start_positions)
                        tokenized_examples["end_positions"].append(end_positions)
                        labels = label_mapper.toidx(labels)
                        tokenized_examples["label"].append(labels)
                    else:                        
                        start_char = answers["answer_start"][0]
                        if type(start_char) is list:
                            start_position, end_position = [], []
                            for sc in start_char:
                                end_char = sc + len(answers["text"][0])
                                sp, ep = search_index(input_ids, sequence_ids, offsets, cls_index, sc, end_char, answer_in_query=answer_in_query, pad_on_right=pad_on_right)
                                start_position.append(sp)
                                end_position.append(ep)
                            tokenized_examples["start_positions"].append(start_position)
                            tokenized_examples["end_positions"].append(end_position)
                        else:
                            end_char = start_char + len(answers["text"][0])
                            start_position, end_position = search_index(input_ids, sequence_ids, offsets, cls_index, start_char, end_char, answer_in_query=answer_in_query, pad_on_right=pad_on_right)
                            tokenized_examples["start_positions"].append(start_position)
                            tokenized_examples["end_positions"].append(end_position)
                        tokenized_examples["label"].append(0)

            for i in range(0, len(tokenized_examples['input_ids'])):
                sample = {'uid': tokenized_examples['id'][i],
                'token_id' : tokenized_examples['input_ids'][i],
                'mask': tokenized_examples['attention_mask'][i],
                'type_id': tokenized_examples['token_type_ids'][i] if "token_type_ids" in tokenized_examples else len(tokenized_examples['input_ids'][i]) * [0],
                'start_position': tokenized_examples['start_positions'][i],
                'end_position': tokenized_examples['end_positions'][i],
                'label': tokenized_examples['label'][i]
                }
                writer.write('{}\n'.format(json.dumps(sample)))

# Validation preprocessing
def prepare_validation_features(tokenizer, samples, output_path, data_type=DataFormat.CLUE_CLASSIFICATION, max_seq_length=384, doc_stride=128, pad_on_right=True, pad_to_max_length=True, label_mapper=None):
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    answer_in_query = True if data_type == DataFormat.CLUE_CLASSIFICATION else False
    if not tokenizer.cls_token:
        # cls_tok_id = tokenizer.eos_token_id
        cls_tok_id = tokenizer.pad_token_id
        prefix_pad = True
    else:
        cls_tok_id = tokenizer.cls_token_id
        prefix_pad = False

    if not tokenizer.sep_token:
        sep_tok_id = tokenizer.eos_token_id
        sep_tok = tokenizer.eos_token
    else:
        sep_tok_id = tokenizer.sep_token_id
        sep_tok = tokenizer.sep_token

    with open(output_path, 'w', encoding='utf-8') as writer:
        for sample in samples:
            context = sample['context']
            if isinstance(context, list):
                context = " {} ".format(sep_tok).join(context)
            question = sample['question']
            answer = sample.get("answer")

            if pad_on_right and prefix_pad:
                question = tokenizer.pad_token + question

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
                verbose=False,
            )

            # Since one example might give us several features if it has a long context, we need a map from a feature to
            # its corresponding example. This key gives us just that.
            sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

            # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
            # corresponding example_id and we will store the offset mappings.
            tokenized_examples["start_positions"] = []
            tokenized_examples["end_positions"] = []
            tokenized_examples["id"] = []
            tokenized_examples["label"] = []
            tokenized_examples["null_ans_index"] = []
            label = None
            offset_mapping = tokenized_examples.pop("offset_mapping")
            # new_offset_mapping = []

            for i in range(len(tokenized_examples["input_ids"])):
                # Grab the sequence corresponding to that example (to know what is the context and what is the question).
                sequence_ids = tokenized_examples.sequence_ids(i)
                if answer_in_query:
                    context_index = 0 if pad_on_right else 1
                else:
                    context_index = 1 if pad_on_right else 0
    
                input_ids = tokenized_examples["input_ids"][i]
                cls_index = input_ids.index(cls_tok_id)
                sep_index = input_ids.index(sep_tok_id)

                # Grab the sequence corresponding to that example (to know what is the context and what is the question).
                sequence_ids = tokenized_examples.sequence_ids(i)
                tokenized_examples["id"].append(sample['id'])
                # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
                # position is part of the context or not.

                if data_type == DataFormat.CLUE_SEQ and "answer" in sample:
                    start_positions, end_positions = [], []
                    labels = ['O'] * len(input_ids)
                    answers = sample["answer"]
                    if len(answers) == 0:
                        start_positions.append(cls_index)
                        end_positions.append(cls_index)
                    else:
                        for aidx, start_char in enumerate(answers["answer_start"]):
                            end_char = start_char + len(answers["text"][aidx])
                            start_position, end_position = search_index(input_ids, sequence_ids, offset_mapping[i], cls_index, start_char, end_char, answer_in_query=answer_in_query, pad_on_right=pad_on_right)
                            start_positions.append(start_position)
                            end_positions.append(end_position)
                            labels = fill_labels(labels, start_position, end_position, label_mapper)
                    tokenized_examples["start_positions"].append(start_positions)
                    tokenized_examples["end_positions"].append(end_positions)
                    labels = label_mapper.toidx(labels)
                    tokenized_examples["label"].append(labels)
                    tokenized_examples["null_ans_index"].append(cls_index)
                else:
                    tokenized_examples["id"].append(sample['id'])
                    if sample['is_impossible'] is not None:
                        label = 1 if sample['is_impossible'] else 0
                        answer['is_impossible'] = sample['is_impossible'] 
                    tokenized_examples["label"].append(label)
                    tokenized_examples["null_ans_index"].append(cls_index)

            # tokenized_examples["offset_mapping"] = offset_mapping
            for i in range(0, len(tokenized_examples['input_ids'])):
                sample = {'uid': tokenized_examples['id'][i],
                'token_id' : tokenized_examples['input_ids'][i],
                'mask': tokenized_examples['attention_mask'][i],
                'type_id': tokenized_examples['token_type_ids'][i] if "token_type_ids" in tokenized_examples else len(tokenized_examples['input_ids'][i]) * [0],
                # 'offset_mapping': tokenized_examples['offset_mapping'][i] if new_offset_mapping else [],
                'offset_mapping':  offset_mapping[i],
                'null_ans_index': tokenized_examples['null_ans_index'][i],
                'context': question if answer_in_query else context,
                'answer': answer,
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
    parser.add_argument('--task_def', type=str, default="experiments/clues/clues_task_def.yml")
    parser.add_argument('--max_seq_length', type=int, default=MAX_SEQ_LEN)
    parser.add_argument('--doc_stride', type=int, default=DOC_STRIDE)
    args = parser.parse_args()
    return args



def main(args):
    # hyper param
    root = args.root_dir
    assert os.path.exists(root)
    suffix = args.model.split('/')[-1]
    literal_model_type = suffix.split('-')[0].upper()

    encoder_model = EncoderModelType[literal_model_type]
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
        from_slow=True,
        revision=args.model_revision
    )
    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"

    if 'uncased' in args.model:
        mt_dnn_suffix = '{}_uncased'.format(mt_dnn_suffix)
    else:
        mt_dnn_suffix = '{}_cased'.format(mt_dnn_suffix)

    mt_dnn_root = os.path.join(root, mt_dnn_suffix)
    if not os.path.isdir(mt_dnn_root):
        os.mkdir(mt_dnn_root)

    task_defs = TaskDefs(args.task_def)

    for task in task_defs.get_task_names():
        task_def = task_defs.get_task_def(task)
        #assert task_def.data_type == DataFormat.CLUE
        logger.info("Task %s" % task)
        for split_name in task_def.split_names:
            print(root)
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
                prepare_train_feature(tokenizer, rows, dump_path, data_type=task_def.data_type, pad_on_right=pad_on_right, label_mapper=task_def.label_vocab, max_seq_length=args.max_seq_length, doc_stride=args.doc_stride)
            else:
                prepare_validation_features(tokenizer, rows, dump_path, data_type=task_def.data_type, pad_on_right=pad_on_right, label_mapper=task_def.label_vocab, max_seq_length=args.max_seq_length, doc_stride=args.doc_stride)



if __name__ == '__main__':
    args = parse_args()
    main(args)
