# coding=utf-8
# Copyright 2018 Google T5 Authors and HuggingFace Inc. team.
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


import os
import tempfile
import unittest

from transformers import is_torch_available

# this line reruns all the tests in BertModelTest; not sure whether this can be prevented
# for now only run module with pytest tests/test_modeling_encoder_decoder.py::EncoderDecoderModelTest
from .test_modeling_bert import BertModelTest
from .utils import require_torch, slow


if is_torch_available():
    from transformers import BertModel, BertForMaskedLM, EncoderDecoderModel


@require_torch
class EncoderDecoderModelTest(unittest.TestCase):
    def prepare_config_and_inputs_bert(self):
        bert_model_tester = BertModelTest.BertModelTester(self)
        encoder_config_and_inputs = bert_model_tester.prepare_config_and_inputs()
        decoder_config_and_inputs = bert_model_tester.prepare_config_and_inputs_for_decoder()
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = encoder_config_and_inputs
        (
            decoder_config,
            decoder_input_ids,
            decoder_token_type_ids,
            decoder_input_mask,
            decoder_sequence_labels,
            decoder_token_labels,
            decoder_choice_labels,
            encoder_hidden_states,
            encoder_attention_mask,
        ) = decoder_config_and_inputs
        return {
            "config": config,
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "decoder_config": decoder_config,
            "decoder_input_ids": decoder_input_ids,
            "decoder_token_type_ids": decoder_token_type_ids,
            "decoder_attention_mask": decoder_input_mask,
            "decoder_sequence_labels": decoder_sequence_labels,
            "decoder_token_labels": decoder_token_labels,
            "decoder_choice_labels": decoder_choice_labels,
            "encoder_hidden_states": encoder_hidden_states,
            "lm_labels": decoder_token_labels,
            "masked_lm_labels": decoder_token_labels,
        }

    def create_and_check_bert_encoder_decoder_model(
        self,
        config,
        input_ids,
        attention_mask,
        encoder_hidden_states,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        **kwargs
    ):
        encoder_model = BertModel(config)
        decoder_model = BertForMaskedLM(decoder_config)
        enc_dec_model = EncoderDecoderModel(encoder_model, decoder_model)
        outputs_encoder_decoder = enc_dec_model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )

        self.assertEqual(outputs_encoder_decoder[0].shape, (decoder_input_ids.shape + (decoder_config.vocab_size,)))
        self.assertEqual(outputs_encoder_decoder[1].shape, (input_ids.shape + (config.hidden_size,)))
        encoder_outputs = (encoder_hidden_states,)
        outputs_encoder_decoder = enc_dec_model(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )

        self.assertEqual(outputs_encoder_decoder[0].shape, (decoder_input_ids.shape + (decoder_config.vocab_size,)))
        self.assertEqual(outputs_encoder_decoder[1].shape, (input_ids.shape + (config.hidden_size,)))

    def create_and_check_bert_encoder_decoder_model_from_pretrained(
        self,
        config,
        input_ids,
        attention_mask,
        encoder_hidden_states,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        **kwargs
    ):
        encoder_model = BertModel(config)
        decoder_model = BertForMaskedLM(decoder_config)
        kwargs = {"encoder_model": encoder_model, "decoder_model": decoder_model}
        enc_dec_model = EncoderDecoderModel.from_pretrained(**kwargs)
        outputs_encoder_decoder = enc_dec_model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )

        self.assertEqual(outputs_encoder_decoder[0].shape, (decoder_input_ids.shape + (decoder_config.vocab_size,)))
        self.assertEqual(outputs_encoder_decoder[1].shape, (input_ids.shape + (config.hidden_size,)))

    def create_and_check_save_and_load_encoder_decoder_model(self, config, decoder_config, **kwargs):
        encoder_model = BertModel(config)
        decoder_model = BertForMaskedLM(decoder_config)
        kwargs = {"encoder_model": encoder_model, "decoder_model": decoder_model}
        enc_dec_model = EncoderDecoderModel.from_pretrained(**kwargs)

        with tempfile.TemporaryDirectory() as temp_dir_name:
            enc_dec_model.save_pretrained(temp_dir_name)
            enc_dec_model.from_pretrained(
                pretrained_model_name_or_path=os.path.join(temp_dir_name, "encoder"),
                decoder_pretrained_model_name_or_path=os.path.join(temp_dir_name, "decoder"),
            )

    def check_loss_output(self, loss):
        self.assertEqual(loss.size(), ())

    def create_and_check_bert_encoder_decoder_model_mlm_labels(
        self,
        config,
        input_ids,
        attention_mask,
        encoder_hidden_states,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        masked_lm_labels,
        **kwargs
    ):
        encoder_model = BertModel(config)
        decoder_model = BertForMaskedLM(decoder_config)
        enc_dec_model = EncoderDecoderModel(encoder_model, decoder_model)
        outputs_encoder_decoder = enc_dec_model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            masked_lm_labels=masked_lm_labels,
        )

        mlm_loss = outputs_encoder_decoder[0]
        self.check_loss_output(mlm_loss)
        # check that backprop works
        mlm_loss.backward()

        self.assertEqual(outputs_encoder_decoder[1].shape, (decoder_input_ids.shape + (decoder_config.vocab_size,)))
        self.assertEqual(outputs_encoder_decoder[2].shape, (input_ids.shape + (config.hidden_size,)))

    def create_and_check_bert_encoder_decoder_model_lm_labels(
        self,
        config,
        input_ids,
        attention_mask,
        encoder_hidden_states,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        lm_labels,
        **kwargs
    ):
        encoder_model = BertModel(config)
        decoder_model = BertForMaskedLM(decoder_config)
        enc_dec_model = EncoderDecoderModel(encoder_model, decoder_model)
        outputs_encoder_decoder = enc_dec_model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            lm_labels=lm_labels,
        )

        lm_loss = outputs_encoder_decoder[0]
        self.check_loss_output(lm_loss)
        # check that backprop works
        lm_loss.backward()

        self.assertEqual(outputs_encoder_decoder[1].shape, (decoder_input_ids.shape + (decoder_config.vocab_size,)))
        self.assertEqual(outputs_encoder_decoder[2].shape, (input_ids.shape + (config.hidden_size,)))

    def create_and_check_bert_encoder_decoder_model_generate(self, input_ids, config, decoder_config, **kwargs):
        encoder_model = BertModel(config)
        decoder_model = BertForMaskedLM(decoder_config)
        enc_dec_model = EncoderDecoderModel(encoder_model, decoder_model)

        # Bert does not have a bos token id, so use pad_token_id instead
        generated_output = enc_dec_model.generate(input_ids, decoder_start_token_id=enc_dec_model.config.pad_token_id)
        self.assertEqual(generated_output.shape, (input_ids.shape[0],) + (decoder_config.max_length,))

    def test_bert_encoder_decoder_model(self):
        input_ids_dict = self.prepare_config_and_inputs_bert()
        self.create_and_check_bert_encoder_decoder_model(**input_ids_dict)

    def test_bert_encoder_decoder_model_from_pretrained(self):
        input_ids_dict = self.prepare_config_and_inputs_bert()
        self.create_and_check_bert_encoder_decoder_model_from_pretrained(**input_ids_dict)

    def test_save_and_load_from_prertained(self):
        input_ids_dict = self.prepare_config_and_inputs_bert()
        self.create_and_check_save_and_load_encoder_decoder_model(**input_ids_dict)

    def test_bert_encoder_decoder_model_mlm_labels(self):
        input_ids_dict = self.prepare_config_and_inputs_bert()
        self.create_and_check_bert_encoder_decoder_model_mlm_labels(**input_ids_dict)

    def test_bert_encoder_decoder_model_lm_labels(self):
        input_ids_dict = self.prepare_config_and_inputs_bert()
        self.create_and_check_bert_encoder_decoder_model_lm_labels(**input_ids_dict)

    def test_bert_encoder_decoder_model_generate(self):
        input_ids_dict = self.prepare_config_and_inputs_bert()
        self.create_and_check_bert_encoder_decoder_model_generate(**input_ids_dict)

    @slow
    def test_real_bert_model_from_pretrained(self):
        model = EncoderDecoderModel.from_pretrained("bert-base-uncased", "bert-base-uncased")
        self.assertIsNotNone(model)
