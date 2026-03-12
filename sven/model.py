import os
import torch
from typing import Optional, Tuple, Union, List
from transformers.cache_utils import DynamicCache
from transformers import AutoTokenizer, AutoConfig, logging
from transformers.modeling_outputs import CausalLMOutputWithPast, CausalLMOutputWithCrossAttentions
from sven.hf import CodeGenForCausalLM, XGLMForCausalLM, GPT2LMHeadCustomModel, GPT2CustomConfig
###
from transformers import LlamaForCausalLM, LlamaConfig, LlamaTokenizer
from transformers.models.qwen2 import Qwen2ForCausalLM, Qwen2Config
from transformers import Starcoder2ForCausalLM, Starcoder2Config
###

class CodeGenPrefixCausalLM(CodeGenForCausalLM):
    def __init__(self, config):
        super().__init__(config)

        self.n_embed_per_head = config.n_embd // config.n_head
        self.prefix_params = torch.nn.ParameterList()
        for _ in range(config.n_control):
            for _ in range(config.n_layer):
                for _ in range(2):
                    param_size = (config.n_head, config.n_prefix_token, self.n_embed_per_head)
                    param = torch.nn.Parameter(torch.zeros(param_size, requires_grad=True))
                    self.prefix_params.append(param)
        self.dropout = torch.nn.Dropout(config.prefix_dropout)

    def get_past_from_prefix(self, control_ids):
        past = list()
        for i in range(self.config.n_layer):
            past.append(list())
            key_stack, val_stack = [], []
            for control_id in control_ids:
                key_idx = control_id * self.config.n_layer * 2 + i * 2
                val_idx = key_idx + 1
                key = self.dropout(self.prefix_params[key_idx])
                val = self.dropout(self.prefix_params[val_idx])
                key_stack.append(key)
                val_stack.append(val)
            past[i].append(torch.stack(key_stack))
            past[i].append(torch.stack(val_stack))
        return past

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)
        else:
            control_ids = [kwargs['control_id']] * input_ids.shape[0]
            past = self.get_past_from_prefix(control_ids)

        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": None,
            "attention_mask": None,
            "token_type_ids": token_type_ids,
        }

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        control_id = None, 
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return super().forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

class IncoderPrefixLM(XGLMForCausalLM):
    def __init__(self, config):
        super().__init__(config)

        self.n_embed_per_head = config.d_model // config.attention_heads
        self.prefix_params = torch.nn.ParameterList()
        for _ in range(config.n_control):
            for _ in range(config.num_layers):
                for _ in range(2):
                    param_size = (config.attention_heads, config.n_prefix_token, self.n_embed_per_head)
                    param = torch.nn.Parameter(torch.zeros(param_size, requires_grad=True))
                    self.prefix_params.append(param)
        self.dropout = torch.nn.Dropout(config.prefix_dropout)

    def get_past_from_prefix(self, control_ids):
        past = list()
        for i in range(self.config.num_layers):
            past.append(list())
            key_stack, val_stack = [], []
            for control_id in control_ids:
                key_idx = control_id * self.config.num_layers * 2 + i * 2
                val_idx = key_idx + 1
                key = self.dropout(self.prefix_params[key_idx])
                val = self.dropout(self.prefix_params[val_idx])
                key_stack.append(key)
                val_stack.append(val)
            past[i].append(torch.stack(key_stack))
            past[i].append(torch.stack(val_stack))
        return past

    def prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None, use_cache=None, **kwargs):
        if past:
            input_ids = input_ids[:, -1:]
        else:
            control_ids = [kwargs['control_id']] * input_ids.shape[0]
            past = self.get_past_from_prefix(control_ids)
        return {
            "input_ids": input_ids,  
            "attention_mask": None,
            "past_key_values": past,
            "use_cache": use_cache,
        }

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        control_id = None, 
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        return super().forward(
            input_ids,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            head_mask,
            cross_attn_head_mask,
            past_key_values,
            inputs_embeds,
            labels,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        )

class SantaPrefixLM(GPT2LMHeadCustomModel):
    def __init__(self, config):
        super().__init__(config)

        self.n_embed_per_head = config.n_embd // config.n_head
        self.prefix_params = torch.nn.ParameterList()
        for _ in range(config.n_control):
            for _ in range(config.n_layer):
                # mha
                for _ in range(2):
                    param_size = (config.n_head, config.n_prefix_token, self.n_embed_per_head)
                    param = torch.nn.Parameter(torch.zeros(param_size, requires_grad=True))
                    self.prefix_params.append(param)
        self.dropout = torch.nn.Dropout(config.prefix_dropout)

    def get_past_from_prefix(self, control_ids):
        past = list()
        for i in range(self.config.n_layer):
            past.append(list())
            key_stack, val_stack = [], []
            for control_id in control_ids:
                key_idx = control_id * self.config.n_layer * 2 + i * 2
                val_idx = key_idx + 1
                key = self.dropout(self.prefix_params[key_idx])
                val = self.dropout(self.prefix_params[val_idx])
                key_stack.append(key)
                val_stack.append(val)
            past[i].append(torch.stack(key_stack))
            past[i].append(torch.stack(val_stack))
        return past

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)
        else:
            control_ids = [kwargs['control_id']] * input_ids.shape[0]
            past = self.get_past_from_prefix(control_ids)

        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": None,
            "attention_mask": None,
            "token_type_ids": token_type_ids,
        }

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        control_id = None, 
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        return super().forward(
            input_ids,
            past_key_values,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            encoder_hidden_states,
            encoder_attention_mask,
            labels,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        )

#####
# llama
class CodeLlamaPrefixLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        num_q_heads = config.num_attention_heads
        num_kv_heads = getattr(config, "num_key_value_heads", num_q_heads)
        head_dim = config.hidden_size // num_q_heads  
        
        model_dtype = next(self.parameters()).dtype if len(list(self.parameters())) > 0 else torch.float32
        
        self.prefix_params = torch.nn.ParameterList()
        for _ in range(config.n_control):
            for _ in range(config.num_hidden_layers):
                for _ in range(2): 
                    param_size = (num_kv_heads, config.n_prefix_token, head_dim)
                    param = torch.nn.Parameter(torch.zeros(param_size, dtype=model_dtype), requires_grad=True)
                    self.prefix_params.append(param)
        self.dropout = torch.nn.Dropout(config.prefix_dropout)

    def get_past_from_prefix(self, control_ids):
        if isinstance(control_ids, torch.Tensor):
            control_ids = control_ids.cpu().tolist()
        elif isinstance(control_ids, int):
            control_ids = [control_ids]
        
        n_layer = self.config.num_hidden_layers
        cache = DynamicCache()
        
        for layer_idx in range(n_layer):
            layer_module = self.model.layers[layer_idx]
            target_device = next(layer_module.parameters()).device
            target_dtype = next(layer_module.parameters()).dtype
            
            layer_keys = []
            layer_vals = []
            for ctrl_id in control_ids:
                base_idx = ctrl_id * n_layer * 2
                k_idx = base_idx + layer_idx * 2
                v_idx = k_idx + 1
                
                layer_keys.append(self.dropout(self.prefix_params[k_idx]).to(device=target_device, dtype=target_dtype))
                layer_vals.append(self.dropout(self.prefix_params[v_idx]).to(device=target_device, dtype=target_dtype))
            
            # stack: [batch, num_kv_heads, prefix_len, head_dim]
            cache.key_cache.append(torch.stack(layer_keys, dim=0))
            cache.value_cache.append(torch.stack(layer_vals, dim=0))
            
        return cache

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        if past_key_values is not None:
            return {
                "input_ids": input_ids[:, -1:],
                "past_key_values": past_key_values,
                "attention_mask": attention_mask,
                "use_cache": True,
            }
            
        control_id = kwargs.get('control_id', 0)
        if isinstance(control_id, torch.Tensor):
            control_id = control_id.item()
        control_ids = [control_id] * input_ids.shape[0]
        past_key_values = self.get_past_from_prefix(control_ids)
            
        if attention_mask is not None:
            prefix_mask = torch.ones(
                attention_mask.shape[0], self.config.n_prefix_token,
                device=attention_mask.device, dtype=attention_mask.dtype
            )
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": True,
        }

    def forward(
        self, input_ids=None, past_key_values=None, attention_mask=None, inputs_embeds=None,
        labels=None, use_cache=None, output_attentions=None, output_hidden_states=None,
        return_dict=None, control_id=None, control_ids=None, **kwargs
    ):
        outputs = super().forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
        
        if not isinstance(outputs, tuple):
            if hasattr(outputs, "logits") and outputs.logits is not None:
                outputs.logits = outputs.logits.float()
            if hasattr(outputs, "loss") and outputs.loss is not None:
                outputs.loss = outputs.loss.float()
                
        return outputs
        
# qwen2
class QwenPrefixLM(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        num_q_heads = config.num_attention_heads
        num_kv_heads = getattr(config, "num_key_value_heads", num_q_heads)
        head_dim = config.hidden_size // num_q_heads
        
        model_dtype = next(self.parameters()).dtype if len(list(self.parameters())) > 0 else torch.float32
        
        self.prefix_params = torch.nn.ParameterList()
        for _ in range(config.n_control):
            for _ in range(config.num_hidden_layers):
                for _ in range(2):
                    param_size = (num_kv_heads, config.n_prefix_token, head_dim)
                    param = torch.nn.Parameter(torch.zeros(param_size, dtype=model_dtype), requires_grad=True)
                    self.prefix_params.append(param)
        self.dropout = torch.nn.Dropout(config.prefix_dropout)

    def get_past_from_prefix(self, control_ids):
        """生成prefix KV cache，返回 DynamicCache 格式"""
        if isinstance(control_ids, torch.Tensor):
            control_ids = control_ids.cpu().tolist()
        elif isinstance(control_ids, int):
            control_ids = [control_ids]
            
        n_layer = self.config.num_hidden_layers
        cache = DynamicCache()
        
        for layer_idx in range(n_layer):
            layer_module = self.model.layers[layer_idx]
            target_device = next(layer_module.parameters()).device
            target_dtype = next(layer_module.parameters()).dtype
            
            layer_keys, layer_vals = [], []
            for ctrl_id in control_ids:
                base_idx = ctrl_id * n_layer * 2
                k_idx = base_idx + layer_idx * 2
                v_idx = k_idx + 1
                layer_keys.append(self.dropout(self.prefix_params[k_idx]).to(device=target_device, dtype=target_dtype))
                layer_vals.append(self.dropout(self.prefix_params[v_idx]).to(device=target_device, dtype=target_dtype))
            
            cache.key_cache.append(torch.stack(layer_keys, dim=0))
            cache.value_cache.append(torch.stack(layer_vals, dim=0))
            
        return cache

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        if past_key_values is not None:
            return {
                "input_ids": input_ids[:, -1:],
                "past_key_values": past_key_values,
                "attention_mask": attention_mask,
                "use_cache": True,
            }
            
        control_id = kwargs.get('control_id', 0)
        if isinstance(control_id, torch.Tensor):
            control_id = control_id.item()
        control_ids = [control_id] * input_ids.shape[0]
        past_key_values = self.get_past_from_prefix(control_ids)
            
        if attention_mask is not None:
            prefix_mask = torch.ones(
                attention_mask.shape[0], self.config.n_prefix_token,
                device=attention_mask.device, dtype=attention_mask.dtype
            )
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": True,
        }

    def forward(
        self, input_ids=None, past_key_values=None, attention_mask=None, inputs_embeds=None,
        labels=None, use_cache=None, output_attentions=None, output_hidden_states=None,
        return_dict=None, control_id=None, control_ids=None, **kwargs
    ):
        outputs = super().forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
        
        if not isinstance(outputs, tuple):
            if hasattr(outputs, "logits") and outputs.logits is not None:
                outputs.logits = outputs.logits.float()
            if hasattr(outputs, "loss") and outputs.loss is not None:
                outputs.loss = outputs.loss.float()
                
        return outputs
    
# starcoder2    
class StarCoder2PrefixLM(Starcoder2ForCausalLM):
    def __init__(self, config: Starcoder2Config):
        super().__init__(config)
        
        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = config.hidden_size // config.num_attention_heads
        
        model_dtype = next(self.parameters()).dtype if len(list(self.parameters())) > 0 else torch.float32
        
        self.prefix_params = torch.nn.ParameterList()
        for _ in range(config.n_control):
            for _ in range(config.num_hidden_layers):
                for _ in range(2): 
                    param_size = (num_kv_heads, config.n_prefix_token, head_dim)
                    param = torch.nn.Parameter(torch.zeros(param_size, dtype=model_dtype), requires_grad=True)
                    self.prefix_params.append(param)
        self.dropout = torch.nn.Dropout(getattr(config, "prefix_dropout", 0.0))

    def get_past_from_prefix(self, control_ids):
        if isinstance(control_ids, torch.Tensor):
            control_ids = control_ids.cpu().tolist()
        elif isinstance(control_ids, int):
            control_ids = [control_ids]
        
        n_layer = self.config.num_hidden_layers
        cache = DynamicCache()
        
        for layer_idx in range(n_layer):
            layer_module = self.model.layers[layer_idx]
            target_device = next(layer_module.parameters()).device
            target_dtype = next(layer_module.parameters()).dtype
                
            layer_keys, layer_vals = [], []  
            for ctrl_id in control_ids:
                base_idx = ctrl_id * n_layer * 2
                k_idx = base_idx + layer_idx * 2
                v_idx = k_idx + 1
                layer_keys.append(self.dropout(self.prefix_params[k_idx]).to(device=target_device, dtype=target_dtype))
                layer_vals.append(self.dropout(self.prefix_params[v_idx]).to(device=target_device, dtype=target_dtype))
                
            cache.key_cache.append(torch.stack(layer_keys, dim=0)) 
            cache.value_cache.append(torch.stack(layer_vals, dim=0))
            
        return cache

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        if past_key_values is not None:
            return {
                "input_ids": input_ids[:, -1:],
                "past_key_values": past_key_values,
                "attention_mask": attention_mask,
                "use_cache": True,
            }

        control_id = kwargs.get('control_id', 0)
        if isinstance(control_id, torch.Tensor):
            control_id = control_id.item()
        control_ids = [control_id] * input_ids.shape[0]
        past_key_values = self.get_past_from_prefix(control_ids)
            
        if attention_mask is not None:
            prefix_mask = torch.ones(
                attention_mask.shape[0], self.config.n_prefix_token,
                device=attention_mask.device, dtype=attention_mask.dtype
            )
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": True,
        }

    def forward(
        self, input_ids=None, past_key_values=None, attention_mask=None, inputs_embeds=None,
        labels=None, use_cache=None, output_attentions=None, output_hidden_states=None,
        return_dict=None, control_id=None, control_ids=None, **kwargs
    ):
        outputs = super().forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
        
        if not isinstance(outputs, tuple):
            if hasattr(outputs, "logits") and outputs.logits is not None:
                outputs.logits = outputs.logits.float()
            if hasattr(outputs, "loss") and outputs.loss is not None:
                outputs.loss = outputs.loss.float()
                
        return outputs
#####

def model_from_pretrained(lm_path, model_type, config):
    kwargs = dict()
    
    ###
    # if any(name in lm_path.lower() for name in ['qwen', 'llama', 'starcoder']):
    #     kwargs['device_map'] = "auto"
    #     kwargs['torch_dtype'] = torch.float32
    ###
    
    if lm_path.startswith('Salesforce/codegen-'):
        if model_type == 'lm':
            model_class = CodeGenForCausalLM
        elif model_type == 'prefix':
            model_class = CodeGenPrefixCausalLM
        else:
            assert False
    elif lm_path.startswith('facebook/incoder-'):
        if config is not None:
            config.attention_dropout = 0.0
            config.dropout = 0.0
        if model_type == 'lm':
            model_class = XGLMForCausalLM
        elif model_type == 'prefix':
            model_class = IncoderPrefixLM
        else:
            assert False
    elif lm_path == 'bigcode/santacoder':
        kwargs['revision'] = 'mha'
        if config is not None:
            config.attn_pdrop = 0.0
            config.embd_pdrop = 0.0
            config.resid_pdrop = 0.0
        if model_type == 'lm':
            model_class = GPT2LMHeadCustomModel
        elif model_type == 'prefix':
            model_class = SantaPrefixLM
        else:
            assert False
    #####
    elif 'qwen2' in lm_path.lower():
        if model_type == 'lm':
            model_class = Qwen2ForCausalLM          # 普通基座
        elif model_type == 'prefix':
            model_class = QwenPrefixLM
        else:
            assert False
    elif 'llama' in lm_path.lower():
        if model_type == 'lm':
            model_class = LlamaForCausalLM          # 普通基座
        elif model_type == 'prefix':
            model_class = CodeLlamaPrefixLM
        else:
            assert False
    elif 'starcoder' in lm_path.lower():
        if model_type == 'lm':
            model_class = Starcoder2ForCausalLM          # 普通基座
        elif model_type == 'prefix':
            model_class = StarCoder2PrefixLM
        else:
            assert False
    #####
    else:
        assert False

    if config is None:
        model = model_class.from_pretrained(lm_path, **kwargs)
    else:
        model = model_class.from_pretrained(lm_path, **kwargs, config=config)

    return model

def config_from_pretrained(lm_path, path):
    if lm_path == 'bigcode/santacoder':
        return GPT2CustomConfig.from_pretrained(path, revision='mha')
    else:
        return AutoConfig.from_pretrained(path)

def save_model(model, path, args):
    prefix_model_types = (
        CodeGenPrefixCausalLM, IncoderPrefixLM, SantaPrefixLM,
        CodeLlamaPrefixLM, QwenPrefixLM, StarCoder2PrefixLM
    )
    if type(model) in prefix_model_types:
        assert args.pretrain_dir.startswith('Salesforce/codegen-') or args.pretrain_dir.startswith('facebook/incoder-') or args.pretrain_dir == 'bigcode/santacoder' or 'llama' in args.pretrain_dir.lower() or 'qwen2' in args.pretrain_dir.lower() or 'starcoder' in args.pretrain_dir.lower()
        config_file = os.path.join(path)
        model.config.save_pretrained(config_file)
        prefix_file = os.path.join(path, 'pytorch_model.bin')
        state_dict = model.prefix_params.state_dict()
        for k, v in state_dict.items():
            state_dict[k] = v.cpu()
        torch.save(state_dict, prefix_file)
        lm_path_file = os.path.join(path, 'lm.txt')
        with open(lm_path_file, 'w') as f:
            f.write(args.pretrain_dir)
    else:
        model.save_pretrained(path)

def load_model(model_type, path, is_training, args):
    logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = tokenizer.bos_token_id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if model_type == 'lm':
        config = config_from_pretrained(path, path)
        model = model_from_pretrained(path, model_type, config)
    elif model_type == 'prefix':
        if is_training:
            lm_path = path
            lm_config = config_from_pretrained(lm_path, lm_path)
            lm_config.n_prefix_token = args.n_prefix_token
            lm_config.prefix_dropout = args.dropout
            lm_config.n_control = 2
            model = model_from_pretrained(lm_path, model_type, lm_config)
        else:
            lm_path_file = os.path.join(path, 'lm.txt')
            assert os.path.exists(lm_path_file)
            with open(lm_path_file) as f:
                lm_path = f.read()
            prefix_config = config_from_pretrained(lm_path, path)
            lm_config = config_from_pretrained(lm_path, lm_path)
            lm_config.n_prefix_token = prefix_config.n_prefix_token
            lm_config.prefix_dropout = prefix_config.prefix_dropout
            lm_config.n_control = prefix_config.n_control
            model = model_from_pretrained(lm_path, model_type, lm_config)
            prefix_file = os.path.join(path, 'pytorch_model.bin')
            model.prefix_params.load_state_dict(torch.load(prefix_file))
    else:
        assert False

    model.resize_token_embeddings(len(tokenizer))
    input_device = parallelize_model(model, args)
    return tokenizer, model, input_device

#####
def parallelize_model(model, args):
    if args.n_gpu > 1:
        if hasattr(model, 'parallelize'): # 针对老模型如 CodeGen
            model.parallelize()
            input_device = model.transformer.first_device
            if hasattr(model, 'prefix_params'):
                model.prefix_params.to(input_device)
        else:
            if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
                input_device = model.model.embed_tokens.weight.device
            elif hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
                input_device = model.transformer.wte.weight.device
            elif hasattr(model, 'transformer') and hasattr(model.transformer, 'word_embeddings'):
                input_device = model.transformer.word_embeddings.weight.device
            else:
                input_device = next(model.parameters()).device
            
            # 将 prefix_params 挂载到输入层同设备
            if hasattr(model, 'prefix_params'):
                model.prefix_params.to(input_device)
    else:
        # 单卡模式
        device = args.device if hasattr(args, 'device') else torch.device("cuda:0")
        model.to(device)
        input_device = device
        if hasattr(model, 'prefix_params'):
            model.prefix_params.to(device)
            
    return input_device
