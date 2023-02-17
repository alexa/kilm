# This code is mostly modified from Huggingface's Bart modeling code:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/bart/modeling_bart.py

from torch.nn import CrossEntropyLoss
from transformers import BartForConditionalGeneration, BartConfig
from transformers.modeling_outputs import (
    Seq2SeqLMOutput,
)
from transformers.utils import logging
from transformers.models.bart.modeling_bart import shift_tokens_right


logger = logging.get_logger(__name__)


class BartForConditionalGenerationWithDesc(BartForConditionalGeneration):
    def __init__(self, config: BartConfig):
        """
        This class is written for structure-based entity-centric pre-training.
        To facilitate the pre-training process by providing more options to 
        deal with token copying loss and the masked lanuage modeling loss, we 
        add two more attributes and one more argument into the `forward` function.
        """
        super().__init__(config)

        self.separate_loss = config.separate_loss
        self.entity_weight = config.entity_weight

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        label_mask=None,
        **kwargs,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.
        label_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for annotating whether the token is a token that is copied from the input sequence (0) or a 
            token that is recovered from a mask token (1). This argument is used when `self.separate_loss` 
            is True.
        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            if self.separate_loss:
                """We calculate the loss for copying the tokens from inputs to outputs 
                independently from the masked language modeling loss."""

                assert label_mask.size() == labels.size()
                loss_ent_fct = CrossEntropyLoss()

                entity_labels = labels.clone()
                labels[label_mask.bool()] = -100
                entity_labels[~label_mask.bool()] = -100

                masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
                masked_ent_loss = loss_ent_fct(lm_logits.view(-1, self.config.vocab_size), entity_labels.view(-1))
                masked_lm_loss = (1-self.entity_weight) * masked_lm_loss + self.entity_weight * masked_ent_loss
            else:
                """We directly calculate the loss over the whole sequence."""
                masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
            

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
