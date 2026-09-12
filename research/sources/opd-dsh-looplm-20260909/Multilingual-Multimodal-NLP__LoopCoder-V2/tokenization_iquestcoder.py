"""Tokenization classes for IQuestCoder."""

import os
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union

import sentencepiece as spm

from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer
from transformers.utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {},
    "tokenizer_file": {},
}
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {}



class IQuestCoderTokenizer(PreTrainedTokenizer):

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token=None,
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        add_bos_token=True,
        add_eos_token=False,
        clean_up_tokenization_spaces=False,
        add_prefix_space=False,
        legacy=None,
        use_default_system_prompt=False,
        chat_template=None,
        **kwargs,
    ):
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
        
        # Legacy behavior handling
        if legacy is None:
            logger.warning_once(
                f"You are using the default legacy behaviour of the {self.__class__.__name__}. This is"
                " expected, and simply means that the `legacy` (previous) behavior will be used so nothing changes for you."
                " If you want to use the new behaviour, set `legacy=False`. This should only be set if you understand what it"
                " means, and thoroughly read the reason why this was added as explained in"
                " https://github.com/huggingface/transformers/pull/24565"
            )
            legacy = True
        
        self.legacy = legacy
        self.vocab_file = vocab_file
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.add_prefix_space = add_prefix_space
        self.use_default_system_prompt = use_default_system_prompt
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)
        

        
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            sp_model_kwargs=self.sp_model_kwargs,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            add_prefix_space=add_prefix_space,
            legacy=legacy,
            use_default_system_prompt=use_default_system_prompt,
            chat_template=chat_template,
            **kwargs,
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(self.vocab_file)

    @property
    def vocab_size(self) -> int:
        """Returns the vocabulary size."""
        return self.sp_model.get_piece_size()

    def get_vocab(self) -> Dict[str, int]:
        """Returns the vocabulary as a dictionary of token to index."""
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize a string.
        
        Args:
            text (`str`): The text to tokenize.
            
        Returns:
            `List[str]`: The list of tokens.
        """
        if self.add_prefix_space:
            text = " " + text
            
        if self.legacy:
            return self.sp_model.encode(text, out_type=str)
        
        # Non-legacy behavior: handle special tokens properly
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token (str) to an id using the vocab."""
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) to a token (str) using the vocab."""
        token = self.sp_model.IdToPiece(index)
        return token

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Converts a sequence of tokens (strings) to a single string.
        
        This method handles special tokens separately to ensure they are not
        decoded using the SentencePiece model.
        
        Args:
            tokens (`List[str]`): The list of tokens to convert.
            
        Returns:
            `str`: The decoded string.
        """
        current_sub_tokens = []
        out_string = ""
        prev_is_special = False
        for i, token in enumerate(tokens):
            # make sure that special tokens are not decoded using sentencepiece model
            if token in self.all_special_tokens:
                if not prev_is_special and i != 0:
                    out_string += " "
                out_string += self.sp_model.decode(current_sub_tokens) + token
                prev_is_special = True
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
                prev_is_special = False
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary and special tokens file to a directory.
        
        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.
            filename_prefix (`str`, *optional*):
                An optional prefix to add to the named of the saved files.
                
        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)

    def build_inputs_with_special_tokens(
        self, 
        token_ids_0: List[int], 
        token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequences for sequence classification tasks by concatenating
        and adding special tokens.
        
        An IQuestCoder sequence has the following format:
        
        - single sequence: `<s> X </s>` (if add_eos_token is True) or `<s> X` (default)
        - pair of sequences: `<s> A </s> <s> B </s>` (if add_eos_token is True) or `<s> A <s> B` (default)
        
        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
                
        Returns:
            `List[int]`: List of input IDs with the appropriate special tokens.
        """
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = bos_token_id + token_ids_0 + eos_token_id

        if token_ids_1 is not None:
            output = output + bos_token_id + token_ids_1 + eos_token_id

        return output

    def get_special_tokens_mask(
        self, 
        token_ids_0: List[int], 
        token_ids_1: Optional[List[int]] = None, 
        already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.
        
        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.
                
        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        bos_token_id = [1] if self.add_bos_token else []
        eos_token_id = [1] if self.add_eos_token else []

        if token_ids_1 is None:
            return bos_token_id + ([0] * len(token_ids_0)) + eos_token_id
        return (
            bos_token_id
            + ([0] * len(token_ids_0))
            + eos_token_id
            + bos_token_id
            + ([0] * len(token_ids_1))
            + eos_token_id
        )

    def create_token_type_ids_from_sequences(
        self, 
        token_ids_0: List[int], 
        token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task.
        
        An IQuestCoder sequence pair mask has the following format:
        
        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```
        
        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).
        
        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
                
        Returns:
            `List[int]`: List of token type IDs according to the given sequence(s).
        """
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = [0] * len(bos_token_id + token_ids_0 + eos_token_id)

        if token_ids_1 is not None:
            output += [1] * len(bos_token_id + token_ids_1 + eos_token_id)

        return output

    @property
    def default_chat_template(self) -> str:
        """
        Returns the default chat template for IQuestCoder.
        
        This template formats conversations with system, user, and assistant roles.
        """
        return DEFAULT_CHAT_TEMPLATE

    def apply_chat_template(
        self,
        conversation: Union[List[Dict[str, str]], "Conversation"],
        chat_template: Optional[str] = None,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_dict: bool = False,
        **tokenizer_kwargs,
    ):
        """
        Apply a chat template to format a conversation.
        
        Args:
            conversation (`List[Dict[str, str]]` or `Conversation`):
                A list of dicts with "role" and "content" keys, representing the conversation history.
            chat_template (`str`, *optional*):
                A Jinja template to use for formatting. If not provided, the tokenizer's default will be used.
            add_generation_prompt (`bool`, *optional*, defaults to `False`):
                Whether to add a generation prompt at the end for the assistant to continue.
            tokenize (`bool`, *optional*, defaults to `True`):
                Whether to tokenize the output. If `False`, returns a string.
            padding (`bool`, *optional*, defaults to `False`):
                Whether to pad sequences.
            truncation (`bool`, *optional*, defaults to `False`):
                Whether to truncate sequences.
            max_length (`int`, *optional*):
                Maximum length of the output.
            return_tensors (`str`, *optional*):
                The type of tensors to return ("pt", "tf", "np", or None).
            return_dict (`bool`, *optional*, defaults to `False`):
                Whether to return a dictionary with additional information.
            **tokenizer_kwargs:
                Additional keyword arguments passed to the tokenizer.
                
        Returns:
            `Union[str, List[int], BatchEncoding]`: The formatted (and optionally tokenized) conversation.
            
        Example:
            ```python
            >>> tokenizer = IQuestCoderTokenizer.from_pretrained("path/to/model")
            >>> conversation = [
            ...     {"role": "system", "content": "You are a helpful assistant."},
            ...     {"role": "user", "content": "Hello!"},
            ...     {"role": "assistant", "content": "Hi there! How can I help you today?"},
            ...     {"role": "user", "content": "What's the weather like?"},
            ... ]
            >>> tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            '<|system|>\\nYou are a helpful assistant.\\n</|system|><|user|>\\nHello!\\n</|user|>...'
            ```
        """
        # Use parent class implementation with our template
        return super().apply_chat_template(
            conversation,
            chat_template=chat_template,
            add_generation_prompt=add_generation_prompt,
            tokenize=tokenize,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            return_dict=return_dict,
            **tokenizer_kwargs,
        )


# Try to import and create Fast tokenizer version
try:
    from transformers import PreTrainedTokenizerFast
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors
    
    class IQuestCoderTokenizerFast(PreTrainedTokenizerFast):
        """
        Construct a "fast" IQuestCoder tokenizer (backed by HuggingFace's *tokenizers* library).
        
        This is a fast implementation of [`IQuestCoderTokenizer`] using the 🤗 Tokenizers library.
        
        Args:
            vocab_file (`str`, *optional*):
                Path to the vocabulary file (SentencePiece model).
            tokenizer_file (`str`, *optional*):
                Path to a tokenizer JSON file.
            unk_token (`str`, *optional*, defaults to `"<unk>"`):
                The unknown token.
            bos_token (`str`, *optional*, defaults to `"<s>"`):
                The beginning of sequence token.
            eos_token (`str`, *optional*, defaults to `"</s>"`):
                The end of sequence token.
            pad_token (`str`, *optional*):
                The token used for padding.
            add_bos_token (`bool`, *optional*, defaults to `True`):
                Whether to add a BOS token at the start of sequences.
            add_eos_token (`bool`, *optional*, defaults to `False`):
                Whether to add an EOS token at the end of sequences.
            add_prefix_space (`bool`, *optional*, defaults to `False`):
                Whether to add an initial space to the input.
            use_default_system_prompt (`bool`, *optional*, defaults to `False`):
                Whether to use the default system prompt.
            chat_template (`str`, *optional*):
                A Jinja template for formatting conversations.
                
        Example:
            ```python
            >>> from tokenization_iquestcoder import IQuestCoderTokenizerFast
            
            >>> tokenizer = IQuestCoderTokenizerFast.from_pretrained("path/to/model")
            >>> tokenizer.encode("Hello, world!")
            [1, 15043, 29892, 3186, 29991]
            ```
        """
        
        vocab_files_names = VOCAB_FILES_NAMES
        pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
        max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
        model_input_names = ["input_ids", "attention_mask"]
        slow_tokenizer_class = IQuestCoderTokenizer
        
        def __init__(
            self,
            vocab_file=None,
            tokenizer_file=None,
            unk_token="<unk>",
            bos_token="<s>",
            eos_token="</s>",
            pad_token=None,
            add_bos_token=True,
            add_eos_token=False,
            add_prefix_space=False,
            use_default_system_prompt=False,
            chat_template=None,
            **kwargs,
        ):
            self.add_bos_token = add_bos_token
            self.add_eos_token = add_eos_token
            self.add_prefix_space = add_prefix_space
            self.use_default_system_prompt = use_default_system_prompt
            self.vocab_file = vocab_file

            if chat_template is None:
                chat_template = DEFAULT_CHAT_TEMPLATE
            
            super().__init__(
                vocab_file=vocab_file,
                tokenizer_file=tokenizer_file,
                unk_token=unk_token,
                bos_token=bos_token,
                eos_token=eos_token,
                pad_token=pad_token,
                add_bos_token=add_bos_token,
                add_eos_token=add_eos_token,
                add_prefix_space=add_prefix_space,
                use_default_system_prompt=use_default_system_prompt,
                chat_template=chat_template,
                **kwargs,
            )
        
        @property
        def can_save_slow_tokenizer(self) -> bool:
            vocab_file = getattr(self, "vocab_file", None)
            return bool(vocab_file) and os.path.isfile(vocab_file)

        def save_vocabulary(
            self, save_directory: str, filename_prefix: Optional[str] = None
        ) -> Tuple[str]:
            if not self.can_save_slow_tokenizer:
                return ()
            if not os.path.isdir(save_directory):
                logger.error(f"Vocabulary path ({save_directory}) should be a directory")
                return ()
            out_vocab_file = os.path.join(
                save_directory,
                (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"],
            )
            if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
                copyfile(self.vocab_file, out_vocab_file)
            return (out_vocab_file,)

        @property
        def default_chat_template(self) -> str:
            """Returns the default chat template."""
            return DEFAULT_CHAT_TEMPLATE
        
        def build_inputs_with_special_tokens(
            self, 
            token_ids_0: List[int], 
            token_ids_1: Optional[List[int]] = None
        ) -> List[int]:
            """Build model inputs with special tokens."""
            bos_token_id = [self.bos_token_id] if self.add_bos_token else []
            eos_token_id = [self.eos_token_id] if self.add_eos_token else []

            output = bos_token_id + token_ids_0 + eos_token_id

            if token_ids_1 is not None:
                output = output + bos_token_id + token_ids_1 + eos_token_id

            return output
        
        def get_special_tokens_mask(
            self, 
            token_ids_0: List[int], 
            token_ids_1: Optional[List[int]] = None, 
            already_has_special_tokens: bool = False
        ) -> List[int]:
            """Retrieve special tokens mask."""
            if already_has_special_tokens:
                return super().get_special_tokens_mask(
                    token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
                )

            bos_token_id = [1] if self.add_bos_token else []
            eos_token_id = [1] if self.add_eos_token else []

            if token_ids_1 is None:
                return bos_token_id + ([0] * len(token_ids_0)) + eos_token_id
            return (
                bos_token_id
                + ([0] * len(token_ids_0))
                + eos_token_id
                + bos_token_id
                + ([0] * len(token_ids_1))
                + eos_token_id
            )
        
        def create_token_type_ids_from_sequences(
            self, 
            token_ids_0: List[int], 
            token_ids_1: Optional[List[int]] = None
        ) -> List[int]:
            """Create token type IDs from sequences."""
            bos_token_id = [self.bos_token_id] if self.add_bos_token else []
            eos_token_id = [self.eos_token_id] if self.add_eos_token else []

            output = [0] * len(bos_token_id + token_ids_0 + eos_token_id)

            if token_ids_1 is not None:
                output += [1] * len(bos_token_id + token_ids_1 + eos_token_id)

            return output

except ImportError:
    # tokenizers library not available, Fast tokenizer not supported
    IQuestCoderTokenizerFast = None
    logger.info(
        "The `tokenizers` library is not installed. "
        "IQuestCoderTokenizerFast will not be available. "
        "Install it with `pip install tokenizers`."
    )

