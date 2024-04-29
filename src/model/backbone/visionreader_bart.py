import random

from transformers.models.mbart.modeling_mbart import (
    MBartEncoder, MBartDecoder,
    MBartModel, MBartForConditionalGeneration,
    MBartConfig,
    shift_tokens_right,
    _expand_mask
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from typing import Any, Dict, Tuple

from transformers.modeling_outputs import ModelOutput, BaseModelOutput, Seq2SeqLMOutput, Seq2SeqModelOutput
from transformers.utils import logging
from vision_module.vision_obj_embedding import VisionObjEmbedding
from vision_module.vision_ocr_embedding import VisionOcrEmbedding
from transformers import AutoModel

logger = logging.get_logger(__name__)

class JointEncoder(MBartEncoder):
    """
    MBartEncoder + visual embedding
    """
    def __init__(self, config, embed_tokens=None):
        super().__init__(config, embed_tokens)
        self.config = config
        self.ocr_embedding = VisionOcrEmbedding(config)
        self.obj_embedding = VisionObjEmbedding(config)
        self.project_vision=nn.Linear(config.d_vision,config.d_model)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,

        ocr_info=None,
        obj_info=None,
        image_features=None,
        vis_attention_mask=None,

        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input_ids)
        inputs_embeds = inputs_embeds + embed_pos

        B, L = inputs_embeds.size()[:-1]

        img_feat = self.project_vision(image_features)
        V_L = img_feat.size(1)

        if self.config.share_vis_lang_layer_norm:
            if (ocr_info and obj_info) is not None:
                ocr_feat = self.ocr_embedding(ocr_info)
                obj_feat = self.obj_embedding(obj_info)
                inputs_embeds = torch.cat([inputs_embeds, img_feat, obj_feat, ocr_feat], dim=1)
                inputs_embeds = self.layernorm_embedding(inputs_embeds)
                V_L +=obj_feat.size(1)
                V_L +=ocr_feat.size(1)
            else:
                inputs_embeds = torch.cat([inputs_embeds, img_feat], dim=1)
                inputs_embeds = self.layernorm_embedding(inputs_embeds)
        else:
            if (ocr_info and obj_info) is not None:
                ocr_feat = self.ocr_embedding(ocr_info)
                obj_feat = self.obj_embedding(obj_info)
                inputs_embeds = self.layernorm_embedding(inputs_embeds)
                inputs_embeds = torch.cat([inputs_embeds, img_feat, obj_feat, ocr_feat], dim=1)
                V_L +=obj_feat.size(1)
                V_L +=ocr_feat.size(1)
            else:
                inputs_embeds = self.layernorm_embedding(inputs_embeds)
                inputs_embeds = torch.cat([inputs_embeds, img_feat], dim=1)


        hidden_states = F.dropout(inputs_embeds, p=self.dropout, training=self.training)

        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id).to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)

        vis_attention_mask = torch.ones(B, V_L, dtype=inputs_embeds.dtype, device=inputs_embeds.device)

        attention_mask = torch.cat([attention_mask,vis_attention_mask], dim=1)
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)


        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if getattr(self.config, "gradient_checkpointing", False):

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                    )
                else:
                    layer_outputs = encoder_layer(hidden_states, attention_mask,
                                                  layer_head_mask= None ,output_attentions=output_attentions)

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )



class VRBartModel(MBartModel):
    def __init__(self, config: MBartConfig):
        super(MBartModel, self).__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        #----- Modified-----#
        # self.encoder = MBartEncoder(config, self.shared)
        self.encoder = JointEncoder(config, self.shared)
        #-------------------#
        self.decoder = MBartDecoder(config, self.shared)

        self.init_weights()

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

        # self.encoder.visual_embedding = self.shared

    def forward(
        self,
        input_ids=None,
        attention_mask=None,

        ocr_info=None,
        obj_info=None,
        image_features=None,
        vis_attention_mask=None,

        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,

        **kwargs,
    ):

        # different to other models, MBart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,

                ocr_info=ocr_info,
                obj_info=obj_info,
                image_features=image_features,
                vis_attention_mask=vis_attention_mask,

                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=False
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id).to(dtype=torch.float, device=input_ids.device)
        B, L = attention_mask.size()
        V_L = encoder_outputs[0].size(1) - L
        if vis_attention_mask is None:
            vis_attention_mask = attention_mask.new_ones(B, V_L)
        encoder_attention_mask = torch.cat([attention_mask, vis_attention_mask], dim=1)

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            # encoder_attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class VRBartForConditionalGeneration(MBartForConditionalGeneration):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"encoder\.version",
        r"decoder\.version",
        r"lm_head\.weight",
    ]

    def __init__(self, config: MBartConfig):
        super(MBartForConditionalGeneration, self).__init__(config)
        self.model = VRBartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,

        ocr_info=None,
        obj_info=None,
        image_features=None,
        vis_attention_mask=None,

        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,

        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,

            ocr_info=ocr_info,
            obj_info=obj_info,
            image_features=image_features,
            vis_attention_mask=vis_attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
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

    def prepare_inputs_for_generation(
        self, decoder_input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        output = {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            "ocr_info": kwargs.get("ocr_info", None),
            "obj_info": kwargs.get("obj_info", None),
            "image_features": kwargs.get("image_features", None),
        }
        if 'vis_attention_mask' in kwargs:
            output['vis_attention_mask'] = kwargs['vis_attention_mask']
        return output
    
    # def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
    #     return self._shift_right(labels)
    
    # def _reorder_cache(self, past, beam_idx):
    #     # if decoder past is not included in output
    #     # speedy decoding is disabled and no need to reorder
    #     if past is None:
    #         logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
    #         return past

    #     reordered_decoder_past = ()
    #     for layer_past_states in past:
    #         # get the correct batch idx from layer past batch dim
    #         # batch dim of `past` is at 2nd position
    #         reordered_layer_past_states = ()
    #         for layer_past_state in layer_past_states:
    #             # need to set correct `past` for each of the four key / value states
    #             reordered_layer_past_states = reordered_layer_past_states + (
    #                 layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
    #             )

    #         assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
    #         assert len(reordered_layer_past_states) == len(layer_past_states)

    #         reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
    #     return reordered_decoder_past
    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: torch.LongTensor = None,
        encoder_outputs: ModelOutput = None,
        **model_kwargs
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1,
                                                                expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(
                0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(
                0, expanded_return_idx)

        if model_kwargs.get("vis_attention_mask", None) is not None:
            model_kwargs['vis_attention_mask'] = model_kwargs['vis_attention_mask'].index_select(
                0, expanded_return_idx)

        if is_encoder_decoder:
            assert encoder_outputs is not None
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_return_idx
            )
            model_kwargs["encoder_outputs"] = encoder_outputs

        return input_ids, model_kwargs