import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from typing import Optional, Union
from transformers import (
    T5PreTrainedModel, T5Config, T5EncoderModel,
    T5GemmaPreTrainedModel, T5GemmaConfig, T5GemmaEncoderModel, T5GemmaModel,
)
from transformers.modeling_outputs import TokenClassifierOutput, Seq2SeqModelOutput, BaseModelOutput
from transformers.utils import (TransformersKwargs, can_return_tuple, auto_docstring)
from typing_extensions import Unpack


class T5ForTokenClassification(T5PreTrainedModel):
    _tied_weights_keys = ["transformer.encoder.embed_tokens.weight"]

    def __init__(self, config: T5Config, dynamic_class_weights=False, smoothing_value=0.0):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.transformer = T5EncoderModel(config)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.transform1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        self.dynamic_class_weights = dynamic_class_weights # binary classification에서 동적으로 class weights를 적용할지 여부

        self.smoothing_value = smoothing_value
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. T5 is a model with relative position embeddings so you
            should be able to pad the inputs on both the right and the left.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for detail.

            [What are input IDs?](../glossary#input-ids)

            To know more on how to prepare `input_ids` for pretraining take a look a [T5 Training](./t5#training).
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.transform1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        logits = self.classifier(hidden_states)

        class_weights = None
        loss = None
        if labels is not None:
            '''if self.num_labels == 1:
                #  We are doing binary classification
                loss_fct = BCEWithLogitsLoss(weight=self.class_weights)
                logits_flatten = logits.view(-1)
                labels_flatten = labels.float().view(-1)
                indices_to_consider = labels != -100
                loss = loss_fct(logits_flatten[indices_to_consider], labels_flatten[indices_to_consider])'''
            if self.dynamic_class_weights:
                num_zero = (labels == 0).int().sum().item()
                num_one = (labels == 1).int().sum().item()
                total = num_zero + num_one
                num_one = max(num_one, 1)  # avoid zero division
                rate_zero = num_zero / total
                
                class_weights = torch.tensor([1, total * rate_zero / num_one])
                class_weights = class_weights.to(logits.device)
            
            loss_fct = CrossEntropyLoss(weight=class_weights, label_smoothing=self.smoothing_value)
            labels = labels.to(logits.device)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits, outputs[2:-1])
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class T5GemmaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, hidden_size: int, num_labels: int, classifier_dropout_rate: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=classifier_dropout_rate)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states

class T5GemmaForTokenClassification(T5GemmaPreTrainedModel):
    def __init__(self, config: T5GemmaConfig, is_encoder_decoder: Optional[bool] = None, smoothing_value=0.0):
        r"""
        is_encoder_decoder (`Optional`, *optional*):
            Whether use encoder_decoder for token classification. When set to False, only encoder is used.
        """
        if is_encoder_decoder is not None:
            config.is_encoder_decoder = is_encoder_decoder
        super().__init__(config)
        self.num_labels = config.num_labels

        if config.is_encoder_decoder:
            self.model = T5GemmaModel(config)
        else:
            self.model = T5GemmaEncoderModel(config)

        hidden_size = config.encoder.hidden_size
        if config.is_encoder_decoder:
            hidden_size = config.decoder.hidden_size

        classifier_dropout = getattr(config, "classifier_dropout_rate", 0.1)
        self.score = T5GemmaClassificationHead(hidden_size, self.num_labels, classifier_dropout)
        
        self.smoothing_value = smoothing_value
        
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        decoder_position_ids: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[BaseModelOutput] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> TokenClassifierOutput:
        r"""
        decoder_position_ids (`torch.LongTensor` of shape `(batch_size, decoder_sequence_length)`, *optional*):
            Indices of positions of each decoder input sequence tokens in the position embeddings. Selected in the range `[0,
            config.decoder.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        if self.config.is_encoder_decoder and (input_ids is None and inputs_embeds is not None):
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__} in encoder-decoder mode."
            )

        if self.config.is_encoder_decoder and (decoder_input_ids is None and decoder_inputs_embeds is None):
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )
            decoder_input_ids = self._shift_right(input_ids)

        if self.config.is_encoder_decoder:
            outputs: Seq2SeqModelOutput = self.model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                decoder_position_ids=decoder_position_ids,
                encoder_outputs=encoder_outputs,
                inputs_embeds=inputs_embeds,
                decoder_inputs_embeds=decoder_inputs_embeds,
                use_cache=False,
                **kwargs,
            )
            last_hidden_state = outputs.last_hidden_state
            hidden_states = outputs.decoder_hidden_states
            attentions = outputs.decoder_attentions
        else:
            outputs: BaseModelOutput = self.model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )
            last_hidden_state = outputs.last_hidden_state
            hidden_states = outputs.hidden_states
            attentions = outputs.attentions

        logits = self.score(last_hidden_state)

        loss = None
        if labels is not None:
            valid_labels = labels.view(-1)
            valid_label_indices = valid_labels.ne(-100)
            valid_labels = valid_labels[valid_label_indices]
            valid_logits = logits.view(-1, self.num_labels)
            valid_logits = valid_logits[valid_label_indices]
            
            loss = self.loss_function(valid_logits, valid_labels, self.config)
    
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
            attentions=attentions,
        )