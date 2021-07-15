# author: Jeiyoon
# baseline code: https://github.com/e0397123/dstc10_metric_track/tree/main/baselines/deep_amfm


from dataclasses import dataclass, field
from typing import Optional

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
    block_span_length: int = field(
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





def main():
    """
    See all possible arguments in src/transformers/training_args.py
    or by passing the --help flag to this script.
    We now keep distinct sets of args, for a cleaner separation of concerns.
    """
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, ))

if __name__ == "__main__":
    main()
