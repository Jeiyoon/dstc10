# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, CTRL, BERT, RoBERTa, XLNet).
GPT, GPT-2 and CTRL are fine-tuned using a causal language modeling (CLM) loss. BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss. XLNet is fine-tuned using a permutation language modeling (PLM) loss.
"""


"""
author: Jeiyoon
baseline code: https://github.com/e0397123/dstc10_metric_track/tree/main/baselines/deep_amfm
"""

import logging
import math
import os
from dataclasses import dataclass, field
from glob import glob
from typing import Optional

from torch.utils.data import ConcatDataset

from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForPermutationLanguageModeling,
    HfArgumentParser,
    LineByLineTextDataset,
    PreTrainedTokenizer,
    TextDataset,
    Trainer,
    TrainingArguments,
    set_seed,
)

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

# dataclass
# https://docs.python.org/ko/3/library/dataclasses.html
@dataclass
class ModelArguments:
    """
    Arguments pretraining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    # Optional: typing 모듈의 Optional은 None이 허용되는 함수의 매개 변수에 대한 타입을 명시할 때 유용합니다.
    # https://www.daleseo.com/python-typing/
    # field: 해당 프로퍼티에 대한 설정을 해줄 수 있다
    # https://sjquant.tistory.com/30
    model_name_or_path: Optional[str] = field(
        default = None,
        metadata = {
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model form scratch."
        },
    )
    model_type: Optional[str] = field(
        default = None,
        metadata = {"help": "If training from scratch, pass a model type from the list"}
    )
    config_name: Optional[str] = field(
        default = None,
        metadata = {"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default = None,
        metadata = {"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    # ???: s3?
    cache_dir: Optional[str] = field(
        default = None,
        metadata = {"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pretraining to what data we are going to input our model for training and eval.
    """
    train_data_file: Optional[str] = field(
        default = None,
        metadata = {"help": "The input training data file (a text file)."}
    )
    # glob format?
    # https://velog.io/@k7120792/Glob-%ED%8C%A8%ED%84%B4%EA%B3%BC-%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D
    train_data_files: Optional[str] = field(
        default = None,
        metadata = {
            "help": "The input training data files (multiple files in glob format). "
            "Very often splitting large files to smaller files can prevent tokenizer going out of memory"
        },
    )
    eval_data_file: Optional[str] = field(
        default = None,
        metadata = {"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    line_by_line: bool = field(
        default = False,
        metadata = {"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    mlm: bool = field(
        default = False,
        metadata = {"help": "Train with masked-language modeling loss instead of language modeling."}
    )
    mlm_probability: float = field(
        default = 0.15,
        metadata = {"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    ### ?
    plm_probability: float = field(
        default = 1 / 6,
        metadata = {"help": "Ratio of length of a span of masked tokens to surrounding context length for permutation language modeling."},
    )
    ### ?
    max_span_length: int = field(
        default = 5,
        metadata = {"help": "Maximum length of a span of masked tokens for permutation language modeling."}
    )
    block_size: int = field(
        default = -1,
        metadata = {
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default = False,
        metadata = {"help": "Overwrite the cached training and evaluation sets"}
    )

def get_dataset(
        args: DataTrainingArguments,
        tokenizer: PreTrainedTokenizer,
        evaluate: bool = False,
        cache_dir: Optional[str] = None,
):
    def _dataset(file_path):
        if args.line_by_line:
            return LineByLineTextDataset(tokenizer = tokenizer,
                                         file_path = file_path,
                                         block_size = args.block_size)
        else:
            return TextDataset(
                tokenizer = tokenizer,
                file_path = file_path,
                block_size = args.block_size,
                overwrite_cache = args.overwrite_cache,
                cache_dir = cache_dir,
            )

    if evaluate:
        return _dataset(args.eval_data_file)
    # glob: glob는 파일들의 리스트를 뽑을 때 사용하는데, 파일의 경로명을 이용해서 입맛대로 요리할 수 있답니다.
    # https://wikidocs.net/83
    elif args.train_data_files:
        return ConcatDataset([_dataset(f) for f in glob(args.train_data_files)])
    else:
        return _dataset(args.train_data_file)

def main():
    """
    See all possible arguments in src/transformers/training_args.py
    or by passing the --help flag to this script.
    We now keep distinct sets of args, for a cleaner separation of concerns.
    """
    # ???: TrainingArguments ?
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    # ???: parser.parse_args_into_dataclasses ?
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file"
            "or remove the --do_eval argument."
        )

    # output_dir : The output directory where the model predictions and checkpoints will be written.
    # overwrite_output_dir:
    # - Overwrite the content of the output directory.
    # - Use this to continue training if output_dir points to a checkpoint directory.
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt = "%m/%d/%Y %H:%M:%S",
        level = logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training
    # The ".form_pretrained" methods guarantee that only one local process can concurrently
    # download model & vocab.
    if model_args.config_name:
        config  = AutoConfig.from_pretrained(model_args.config_name,
                                             cache_dir = model_args.cache_dir)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path,
                                            cache_dir = model_args.cache_dir)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name,
                                                  cache_dir = model_args.cache_dir)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path,
                                                  cache_dir = model_args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch."
            "This is not supported, but you can do it from another script, save it."
            "and load it from here, using --tokenizer_name"
        )

    if model_args.model_name_or_path:
        model = AutoModelWithLMHead.from_pretrained(
            model_args.model_name_or_path,
            from_tf = bool(".ckpt" in model_args.model_name_or_path),
            config = config,
            cache_dir = model_args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelWithLMHead.from_config(config)

    # ???: why?
    model.resize_token_embeddings(len(tokenizer))

    if config.model_type in ["bert", "roberta", "distilbert", "camembert"] and not data_args.mlm:
        raise ValueError(
            "BERT and RoBERTa-like models do not have LM heads but masked LM heads."
            "They must be run using the --mlm flag (masked language modeling)."
        )

    # ???: why?
    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.max_len)

    """
    Get datasets
    """
    train_dataset = (
        get_dataset(data_args,
                    tokenizer = tokenizer, cache_dir = model_args.cache_dir) if training_args.do_train else None
    )

    eval_dataset = (
        get_dataset(data_args, tokenizer = tokenizer, evaluate = True, cache_dir = model_args.cache_dir)
        if training_args.do_eval else None
    )

    if config.model_type == "xlnet":
        data_collator = DataCollatorForPermutationLanguageModeling(
            tokenizer = tokenizer,
            plm_probability = data_args.plm_probability,
            max_span_length = data_args.max_span_length,
        )
    else:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer = tokenizer, mlm = data_args.mlm, mlm_probability = data_args.mlm_probability
        )

    """
    Initialize our Trainer
    """
    trainer = Trainer(
        model = model,
        args = training_args,
        data_collator = data_collator,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        prediction_loss_only = True,
    )

    """
    Training
    """
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
            else None
        )
        trainer.train(model_path = model_path)
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    """
    Evaluation
    """
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        perplexity = math.exp(eval_output["eval_loss"])
        result = {"perplexity": perplexity}

        output_eval_files = os.path.join(training_args.output_dir, "eval_results_lm.txt")

        if trainer.is_world_master():
            with open(output_eval_files, "w") as writer:
                logger.info("***** Eval results *****")

                for key in sorted(result.keys()):
                    logger.info("   %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

        results.update(result)

    return results

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ == "__main__":
    main()
