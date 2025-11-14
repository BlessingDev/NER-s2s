
from transformers import GenerationMixin, Gemma3PreTrainedModel, Gemma3TextConfig, Gemma3TextModel
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.utils import auto_docstring
from transformers.cache_utils import Cache
from typing import Optional, Union
from transformers.utils.generic import can_return_tuple
import torch
import torch.nn as nn

def mask_logits(allowed_index, score_tensor):
    masked_tensor = torch.ones_like(score_tensor).to(score_tensor.device) * -1e9
    for idx in allowed_index:
        masked_tensor[idx] = score_tensor[idx]
    return masked_tensor

def inerd_decoding(current_score: torch.Tensor, sentence_input_ids: list, entity_list_ids: set, eos_token_id: int, ct_token_id: int, es_token_id: int, tcs_token_id: int, space_token_id: int, generated_ids=None):
    # current_score: (vocab_size,)
    # input_ids: (seq_len)
    # generated_ids: (gen_len)

    previous_token_id = ct_token_id
    generating_entity = False
    if len(generated_ids) > 0:
        previous_token_id = generated_ids[-1]
    
        # generation 여부 판별
        for idx in range(len(generated_ids)-1, -1, -1):
            if generated_ids[idx] == es_token_id:
                break
            elif generated_ids[idx] == tcs_token_id:
                generating_entity = True
                break

    if previous_token_id == eos_token_id:
        return current_score
    
    # 여기서부터 어휘 사전 점수 조절
    if previous_token_id == ct_token_id or previous_token_id == es_token_id:
        allowed_index = entity_list_ids
        allowed_index.add(eos_token_id)
        current_score = mask_logits(allowed_index, current_score)
    elif previous_token_id in entity_list_ids and not generating_entity:
        allowed_index = entity_list_ids
        allowed_index.add(tcs_token_id)
        current_score = mask_logits(allowed_index, current_score)
    elif generating_entity:
        if previous_token_id == tcs_token_id:
            allowed_index = set(sentence_input_ids)
            current_score = mask_logits(allowed_index, current_score)
        else:
            following_token_in_sentence_indices = [sentence_input_ids[i + 1] for i, id in enumerate(sentence_input_ids) if id == previous_token_id and i + 1 < len(sentence_input_ids)]
            allowed_index = set(following_token_in_sentence_indices)
            allowed_index.add(space_token_id)
            allowed_index.add(es_token_id)
            current_score = mask_logits(allowed_index, current_score)
    
    return current_score

class Gemma3ForCausalLM(Gemma3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    config: Gemma3TextConfig
    base_model_prefix = "language_model"

    def __init__(self, config: Gemma3TextConfig):
        super().__init__(config)
        self.model = Gemma3TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def initialize_inerd(self, 
        line_break_indicator: list,
        start_of_turn_id: int,
        end_of_turn_id: int,
        ct_token_id: int,
        space_token_id: int, 
        es_token_id: int,
        tcs_token_id: int):
        self.start_of_turn_id = start_of_turn_id
        self.end_of_turn_id = end_of_turn_id
        self.ct_token_id = ct_token_id
        self.space_token_id = space_token_id
        self.es_token_id = es_token_id
        self.tcs_token_id = tcs_token_id
        
        self.line_break_indicator = " ".join([str(item) for item in line_break_indicator if item != space_token_id])

    def initialize_inerd_batch(self, batch_input_ids):
        batch_size = batch_input_ids.size(0)
        self.sentence_ids = []
        self.entity_list_ids = []
        
        for batch_idx in range(batch_size):
            input_ids_list = batch_input_ids[batch_idx].tolist()
            input_ids_str = " ".join([str(id) for id in input_ids_list])
            
            input_ids_str = input_ids_str.split(' ' + str(self.start_of_turn_id) + ' ')[1]
            input_ids_str = input_ids_str.split(' ' + str(self.end_of_turn_id) + ' ')[0]
            input_ids_line_list = input_ids_str.split(' ' + self.line_break_indicator + ' ')
            
            sentence_ids = input_ids_line_list[-1].strip().split(" ")
            self.sentence_ids.append([int(id) for id in sentence_ids if id != ''])
            
            entity_list_ids = input_ids_line_list[-2].strip().split(" ")
            entity_list_ids = set([int(id) for id in entity_list_ids if id != ''])
            entity_list_ids.add(self.space_token_id)
            self.entity_list_ids.append(set(entity_list_ids))

        self.generated_ids = [list() for _ in range(batch_size)]
    
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, Gemma3ForCausalLM

        >>> model = Gemma3ForCausalLM.from_pretrained("google/gemma-2-9b")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")

        >>> prompt = "What is your favorite condiment?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "What is your favorite condiment?"
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        # inerd decoding 적용
        batch_size = logits.size(0)
        seq_len = len(self.generated_ids[0])
        '''if seq_len > 400:
            # 너무 긴 시퀀스는 논리 오류 가능성 있음
            print("Warning: seq_len > 400 during inerd decoding.")'''
        
        for batch_idx in range(batch_size):
            if seq_len > 0:
                lm_logits_batch = logits[batch_idx]
                # 입력 문장만 추출
                
                sentence_ids = self.sentence_ids[batch_idx].copy()
                entity_list_ids = self.entity_list_ids[batch_idx].copy()

                current_score = lm_logits_batch[-1]
                
                masked_score = inerd_decoding(
                    current_score,
                    sentence_ids,
                    entity_list_ids,
                    self.end_of_turn_id,
                    self.ct_token_id,
                    self.es_token_id,
                    self.tcs_token_id,
                    self.space_token_id,
                    self.generated_ids[batch_idx]
                )
                logits[batch_idx][-1] = masked_score

            self.generated_ids[batch_idx].append(torch.argmax(logits[batch_idx][-1]).item())
        
        # 추가한 코드 끝
        
        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
