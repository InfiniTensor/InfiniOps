graph():
    %l_input_ids_ : torch.Tensor [num_users=1] = placeholder[target=l_input_ids_]
    %s72 : torch.SymInt [num_users=2] = placeholder[target=s72]
    %l_self_modules_embed_tokens_parameters_weight_ : torch.nn.parameter.Parameter [num_users=1] = placeholder[target=l_self_modules_embed_tokens_parameters_weight_]
    %l_self_modules_layers_modules_0_modules_input_layernorm_parameters_weight_ : torch.nn.parameter.Parameter [num_users=1] = placeholder[target=l_self_modules_layers_modules_0_modules_input_layernorm_parameters_weight_]
    %l_self_modules_layers_modules_0_modules_self_attn_modules_qkv_proj_parameters_weight_ : vllm.model_executor.parameter.ModelWeightParameter [num_users=1] = placeholder[target=l_self_modules_layers_modules_0_modules_self_attn_modules_qkv_proj_parameters_weight_]
    %l_self_modules_layers_modules_0_modules_self_attn_modules_qkv_proj_parameters_bias_ : torch.nn.parameter.Parameter [num_users=1] = placeholder[target=l_self_modules_layers_modules_0_modules_self_attn_modules_qkv_proj_parameters_bias_]
    %l_self_modules_layers_modules_0_modules_self_attn_modules_rotary_emb_buffers_cos_sin_cache_ : torch.Tensor [num_users=1] = placeholder[target=l_self_modules_layers_modules_0_modules_self_attn_modules_rotary_emb_buffers_cos_sin_cache_]
    %l_positions_ : torch.Tensor [num_users=1] = placeholder[target=l_positions_]
    %s80 : torch.SymInt [num_users=0] = placeholder[target=s80]
    %long : [num_users=1] = call_method[target=long](args = (%l_input_ids_,), kwargs = {})
    %embedding : [num_users=2] = call_function[target=torch.nn.functional.embedding](args = (%long, %l_self_modules_embed_tokens_parameters_weight_), kwargs = {})
    %infini_rms_norm : [num_users=1] = call_function[target=torch.ops.vllm.infini_rms_norm](args = (%embedding, %l_self_modules_layers_modules_0_modules_input_layernorm_parameters_weight_, 1e-06), kwargs = {})
    %infini_unquantized_gemm : [num_users=1] = call_function[target=torch.ops.vllm.infini_unquantized_gemm](args = (%infini_rms_norm, %l_self_modules_layers_modules_0_modules_self_attn_modules_qkv_proj_parameters_weight_, %l_self_modules_layers_modules_0_modules_self_attn_modules_qkv_proj_parameters_bias_), kwargs = {})
    %view : [num_users=1] = call_method[target=view](args = (%infini_unquantized_gemm, (%s72, 2560)), kwargs = {})
    %split : [num_users=3] = call_method[target=split](args = (%view, [2048, 256, 256]), kwargs = {dim: -1})
    %getitem : [num_users=1] = call_function[target=operator.getitem](args = (%split, 0), kwargs = {})
    %getitem_1 : [num_users=1] = call_function[target=operator.getitem](args = (%split, 1), kwargs = {})
    %getitem_2 : [num_users=1] = call_function[target=operator.getitem](args = (%split, 2), kwargs = {})
    %to : [num_users=1] = call_method[target=to](args = (%l_self_modules_layers_modules_0_modules_self_attn_modules_rotary_emb_buffers_cos_sin_cache_, torch.float16), kwargs = {})
    %infini_rotary_embedding_v2 : [num_users=2] = call_function[target=torch.ops.vllm.infini_rotary_embedding_v2](args = (%l_positions_, %getitem, %getitem_1, %to, 128, True), kwargs = {})
    %getitem_3 : [num_users=1] = call_function[target=operator.getitem](args = (%infini_rotary_embedding_v2, 0), kwargs = {})
    %getitem_4 : [num_users=1] = call_function[target=operator.getitem](args = (%infini_rotary_embedding_v2, 1), kwargs = {})
    %size : [num_users=1] = call_function[target=torch.Size](args = ([%s72, 2048],), kwargs = {})
    %empty : [num_users=1] = call_function[target=torch.empty](args = (%size,), kwargs = {dtype: torch.float16, device: npu:0})
    %view_1 : [num_users=1] = call_method[target=view](args = (%getitem_3, -1, 16, 128), kwargs = {})
    %view_2 : [num_users=1] = call_method[target=view](args = (%empty, -1, 16, 128), kwargs = {})
    %view_3 : [num_users=1] = call_method[target=view](args = (%getitem_4, -1, 2, 128), kwargs = {})
    %view_4 : [num_users=1] = call_method[target=view](args = (%getitem_2, -1, 2, 128), kwargs = {})
    return (view_1, view_3, view_4, view_2, embedding)

# --- node count summary ---
     9  <placeholder>
     8  <call_method>
     5  <built-in function getitem>
     1  <function embedding at 0xfffe4b5a2980>
    %view_3 : [num_users=1] = call_method[target=view](args = (%getitem_4, -1, 2, 128), kwargs = {})
    %view_4 : [num_users=1] = call_method[target=view](args = (%getitem_2, -1, 2, 128), kwargs = {})
    return (view_1, view_3, view_4, view_2, embedding)

# --- node count summary ---
     9  <placeholder>
     8  <call_method>
     5  <built-in function getitem>
     1  <function embedding at 0xfffe4b5a2980>
     1  vllm.infini_rms_norm
     1  vllm.infini_unquantized_gemm
     1  vllm.infini_rotary_embedding_v2
     1  <class 'torch.Size'>
     1  <built-in method empty of type object at 0xfffe5f1fbb50>
     1  <output>
