from typing import Any, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import AutoModel
from transformers import Blip2Processor, Blip2ForConditionalGeneration, AutoConfig


# class LMWrapper(nn.Module):
#     def __init__(self, nn):
#         self.nn = nn    

class BLIPWrapper(nn.Module):
    def __init__(self, args):
        super().__init__()
    
        # config = AutoConfig.from_pretrained(args.model_card)    

        self.config = AutoConfig.from_pretrained(args.model_card)
        model = Blip2ForConditionalGeneration.from_pretrained(args.model_card) 
        
        model.config.max_length = 50

        self.vision_model = model.vision_model  # nn.Modle
        self.query_tokens = model.query_tokens
        
        self.qformer = model.qformer

        # self.vision_model.eval()
        # self.vision_model.requires_grad_(False)
        # for param in self.vision_model.parameters():
        #     param.requires_grad = False

        # self.qformer.eval()
        # self.qformer.requires_grad_(False)
        # for param in self.qformer.parameters():
        #     param.requires_grad = False

        # self.query_tokens.requires_grad = False
        # for param in self.query_tokens.parameters():
        #     param.requires_grad = False
        # for param in self.vision_model.parameters():
        #     param.requires_grad = False
        # for param in self.qformer.parameters():
        #     param.requires_grad = False

        self.language_projection = model.language_projection        
        self.language_model = model.language_model
        

        for param in self.language_projection.parameters():
            param.requires_grad = False
        for param in self.language_model.parameters():
            param.requires_grad = False

        del model

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()
        
    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        """
        Overrides `generate` function to be able to use the model as a conditional generator.

        Args:
            pixel_values (`torch.FloatTensor` of shape (batch_size, num_channels, height, width)):
                Input images to be processed.
            input_ids (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                The sequence used as a prompt for the generation.
            attention_mask (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                Mask to avoid performing attention on padding token indices

        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """

        batch_size = pixel_values.shape[0]
        image_embeds = self.vision_model(pixel_values, return_dict=True).last_hidden_state
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        query_output = query_outputs.last_hidden_state

        language_model_inputs = self.language_projection(query_output)
        language_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )
        if input_ids is None:
            input_ids = (
                torch.LongTensor([[self.config.text_config.bos_token_id]])
                .repeat(batch_size, 1)
                .to(image_embeds.device)
            )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = torch.cat([language_attention_mask, attention_mask.to(language_attention_mask.device)], dim=1)

        # concatenate query embeddings with prompt embeddings
        inputs_embeds = self.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)

        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        return outputs

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ):
    
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # return_dict = None
        # print(self.config.use_return_dict) -> True

        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_embeds = vision_outputs[0]

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
        # image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long)  

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        query_output = query_outputs[0]

        # step 3: use the language model, conditioned on the query outputs and the prompt
        language_model_inputs = self.language_projection(query_output)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )
        inputs_embeds = language_model_inputs

        # if attention_mask is None:
        #     attention_mask = torch.ones_like(input_ids)

        expected_device = language_model_attention_mask.device
        # attention_mask = torch.cat([language_model_attention_mask, attention_mask.to(expected_device)], dim=1)
        attention_mask = language_model_attention_mask

        # if self.config.use_decoder_only_language_model: 
        if False:
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            # logits = outputs.logits if return_dict else outputs[0]
            logits = outputs[0]
            loss = None
            # we compute the loss here since we need to take into account the sequence length of the query embeds
            if labels is not None:
                # labels = labels.to(logits.device)
                # logits = logits[:, -labels.size(1) :, :]
                # Shift so that tokens < n predict n
                # shift_logits = logits[..., :-1, :].contiguous()
                # shift_labels = labels[..., 1:].contiguous().to(logits.device)

                # Flatten the tokens
                loss_fct = CrossEntropyLoss(reduction="mean")

                # loss = loss_fct(shift_logits.view(-1, self.config.text_config.vocab_size), shift_labels.view(-1))
                loss = loss_fct(logits.view(-1, self.config.text_config.vocab_size), labels.view(-1))
                return loss
        else:
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=labels,
            )
            loss = outputs.loss if return_dict else outputs[0]
            logits = outputs.logits if return_dict else outputs[1]

            return loss

        if not return_dict:
            output = (logits, vision_outputs, query_outputs, outputs)
            return ((loss,) + output) if loss is not None else output

        # return Blip2ForConditionalGenerationModelOutput(

