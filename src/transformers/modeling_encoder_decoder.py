
# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
""" Classes to support Encoder-Decoder architectures """


import logging
from typing import Optional

from .modeling_auto import AutoModel, AutoModelWithLMHead
from .modeling_utils import PreTrainedModel


logger = logging.getLogger(__name__)


class EncoderDecoderModel(PreTrainedModel):
    r"""
        :class:`~transformers.EncoderDecoder` is a generic model class that will be
        instantiated as a transformer architecture with one of the base model
        classes of the library as encoder and another one as
        decoder when created with the `AutoModel.from_pretrained(pretrained_model_name_or_path)`
        class method for the encoder and `AutoModelWithLMHead.from_pretrained(pretrained_model_name_or_path)` class method for the decoder.
    """
    config_class = EncoderDecoderConfig

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        encoder: Optional[PreTrainedModel] = None,
        decoder: Optional[PreTrainedModel] = None,
    ):
        assert config is not None or (
            encoder is not None and decoder is not None
        ), "Either a configuration or an Encoder and a decoder has to be provided"
        if config is None:
            config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config)
        else:
            assert isinstance(config, self.config_class), "config: {} has to be of type {}".format(
                config, self.config_class
            )
        # initialize with config
        super().__init__(config)

        if encoder is None:
            from transformers import AutoModel

            encoder = AutoModel.from_config(config.encoder)

        if decoder is None:
            from transformers import AutoModelWithLMHead

            decoder = AutoModelWithLMHead.from_config(config.decoder)

    def __init__(self, encoder, decoder):
        assert encoder is not None, "The encoder has to be defined"
        assert decoder is not None, "The encoder has to be defined"

        # TODO: think about how to handle the config
        config = self._init_config(encoder.config, decoder.config)
        super().__init__(config)

        import ipdb
        ipdb.set_trace()

        self.encoder = encoder
        assert self.encoder.get_output_embeddings() is None, "The encoder {} should not have a LM Head. Please use a model without LM Head"
        self.decoder = decoder
        self.is_encoder_decoder = True

    def _init_config(self, encoder_config, decoder_config):
        # TODO: correct the function here
        # Seq-2-Seq should have at least same word embeddings for encoder and decoder
        # so all special tokens should be the same
        assert encoder_config.pad_token_id == decoder_config.pad_token_id
        assert encoder_config.bos_token_id == decoder_config.bos_token_id
        assert encoder_config.eos_token_id == decoder_config.eos_token_id
        assert encoder_config.vocab_size == decoder_config.vocab_size

        return decoder_config

    def get_encoder(self):
        return self.encoder

    @classmethod
    def from_encoder_decoder_pretrained(
        cls,
        pretrained_model_name_or_path=None,
        decoder_pretrained_model_name_or_path=None,
        *model_args,
        **kwargs
    ) -> PreTrainedModel:
        r""" Instantiates an encoder and a decoder from one or two base classes of the library from pre-trained model checkpoints.


        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated).
        To train the model, you need to first set it back in training mode with `model.train()`.

        Params:
            encoder_pretrained_model_name_or_path (:obj: `str`, `optional`, defaults to `None`):
                information necessary to initiate the encoder. Either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/encoder``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            decoder_pretrained_model_name_or_path (:obj: `str`, `optional`, defaults to `None`):
                information necessary to initiate the decoder. Either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/decoder``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method

            kwargs: (`optional`) Remaining dictionary of keyword arguments.
                Can be used to update the configuration object (after it being loaded) and initiate the model. (e.g. ``output_attention=True``). Behave differently depending on whether a `config` is provided or automatically loaded:

        Examples::

            from tranformers import EncoderDecoder

            model = EncoderDecoder.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased') # initialize Bert2Bert
        """

        # keyword arguments come in 3 flavors: encoder-specific (prefixed by
        # `encoder_`), decoder-specific (prefixed by `decoder_`) and those
        # that apply to the model as a whole.
        # We let the specific kwargs override the common ones in case of conflict.

        kwargs_encoder = {
            argument: value
            for argument, value in kwargs.items()
            if not argument.startswith("decoder_")
        }

        kwargs_decoder = {
            argument[len("decoder_") :]: value
            for argument, value in kwargs.items()
            if argument.startswith("decoder_")
        }

        # Load and initialize the encoder and decoder
        # The distinction between encoder and decoder at the model level is made
        # by the value of the flag `is_decoder` that we need to set correctly.
        encoder = kwargs_encoder.pop("model", None)
        if encoder is None:
            assert pretrained_model_name_or_path is not None, "If `model` is not defined as an argument, a `encoder_pretrained_model_name_or_path` has to be defined"
            encoder = AutoModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs_encoder)
        encoder.config.is_decoder = False

        decoder = kwargs_decoder.pop("model", None)
        if decoder is None:
            assert decoder_pretrained_model_name_or_path is not None, "If `decoder_model` is not defined as an argument, a `decoder_pretrained_model_name_or_path` has to be defined"

            # TODO: Maybe make two classes 1) AutoModel 2) AutoModelWithLMHead

            decoder = AutoModelWithLMHead.from_pretrained(decoder_pretrained_model_name_or_path, **kwargs_decoder)
        decoder.config.is_decoder = True

        model = cls(encoder=encoder, decoder=decoder)

        return model

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        head_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_head_mask=None,
        decoder_inputs_embeds=None,
        masked_lm_labels=None,
        lm_labels=None,
        **kwargs,
    ):

        """
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary for the encoder.
                Indices can be obtained using :class:`transformers.PretrainedTokenizer`.
                See :func:`transformers.PreTrainedTokenizer.encode` and
                :func:`transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
                Mask to avoid performing attention on padding token indices for the encoder.
                Mask values selected in ``[0, 1]``:
                ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
            head_mask: (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
                Mask to nullify selected heads of the self-attention modules for the encoder.
                Mask values selected in ``[0, 1]``:
                ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
            encoder_outputs (:obj:`tuple(tuple(torch.FloatTensor)`, `optional`, defaults to :obj:`None`):
                Tuple consists of (`last_hidden_state`, `optional`: `hidden_states`, `optional`: `attentions`)
                `last_hidden_state` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`) is a sequence of hidden-states at the output of the last layer of the encoder.
                Used in the cross-attention of the decoder.
            decoder_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`, defaults to :obj:`None`):
                Provide for sequence to sequence training to the decoder.
                Indices can be obtained using :class:`transformers.PretrainedTokenizer`.
                See :func:`transformers.PreTrainedTokenizer.encode` and
                :func:`transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
            decoder_attention_mask (:obj:`torch.BoolTensor` of shape :obj:`(batch_size, tgt_seq_len)`, `optional`, defaults to :obj:`None`):
                Default behavior: generate a tensor that ignores pad tokens in decoder_input_ids. Causal mask will also be used by default.
            decoder_head_mask: (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
                Mask to nullify selected heads of the self-attention modules for the decoder.
                Mask values selected in ``[0, 1]``:
                ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
            decoder_inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, target_sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
                Optionally, instead of passing :obj:`decoder_input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `decoder_input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            masked_lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
                Labels for computing the masked language modeling loss for the decoder.
                Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
                Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
                in ``[0, ..., config.vocab_size]``
            lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
                Labels for computing the left-to-right language modeling loss (next word prediction) for the decoder.
                Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
                Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
                in ``[0, ..., config.vocab_size]``
            kwargs: (`optional`) Remaining dictionary of keyword arguments. Keyword arguments come in two flavors:
                - Without a prefix which will be input as `**encoder_kwargs` for the encoder forward function.
                - With a `decoder_` prefix which will be input as `**decoder_kwargs` for the decoder forward function.

        # If the root output directory does not exist, create it
        if not os.path.exists(save_directory):
            os.mkdir(save_directory)

        # Check whether the output directory is empty or not
        sub_directories = [
            directory
            for directory in os.listdir(save_directory)
            if os.path.isdir(os.path.join(save_directory, directory))
        ]

        if len(sub_directories) > 0:
            if "encoder" in sub_directories and "decoder" in sub_directories:
                print(
                    "WARNING: there is an older version of encoder-decoder saved in"
                    + " the output directory. The default behaviour is to overwrite them."
                )

            # Empty the output directory
            for directory_to_remove in sub_directories:
                # Remove all files into the subdirectory
                files_to_remove = os.listdir(os.path.join(save_directory, directory_to_remove))
                for file_to_remove in files_to_remove:
                    os.remove(os.path.join(save_directory, directory_to_remove, file_to_remove))
                # Remove the subdirectory itself
                os.rmdir(os.path.join(save_directory, directory_to_remove))

            assert len(os.listdir(save_directory)) == 0  # sanity check

        # Create the "encoder" directory inside the output directory and save the encoder into it
        if not os.path.exists(os.path.join(save_directory, "encoder")):
            os.mkdir(os.path.join(save_directory, "encoder"))
        self.encoder.save_pretrained(os.path.join(save_directory, "encoder"))

        # Create the "encoder" directory inside the output directory and save the decoder into it
        if not os.path.exists(os.path.join(save_directory, "decoder")):
            os.mkdir(os.path.join(save_directory, "decoder"))
        self.decoder.save_pretrained(os.path.join(save_directory, "decoder"))

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        lm_labels=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        head_mask=None,
    ):

        """
        Params:
            encoder_input_ids: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``
                Indices of encoder input sequence tokens in the vocabulary.
            decoder_input_ids: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``
                Indices of decoder input sequence tokens in the vocabulary.
            kwargs: (`optional`) Remaining dictionary of keyword arguments.
        """

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, head_mask=head_mask
            )

        hidden_states = encoder_outputs[0]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            inputs_embeds=decoder_inputs_embeds,
            attention_mask=attention_mask,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=head_mask
        )

        return decoder_outputs + encoder_outputs

    def prepare_inputs_for_generation(self, input_ids, past, attention_mask, **kwargs):
        assert past is not None, "past has to be defined for encoder_outputs"

        # first step
        if type(past) is tuple:
            encoder_outputs = past
        else:
            encoder_outputs = (past,)

        decoder_inputs = self.decoder.prepare_inputs_for_generation(input_ids, attention_mask)

        return {
            "decoder_input_ids": decoder_inputs['input_ids'],
            "encoder_outputs": encoder_outputs,
            "attention_mask": decoder_inputs['attention_mask'],
        }

    def _reorder_cache(self, past, beam_idx):
        # as a default encoder-decoder models do not re-order the past. TODO: might
        # have to be updated, e.g. if GPT2 is to be used as a decoder
        return past
