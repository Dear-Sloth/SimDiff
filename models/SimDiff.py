    
import torch
import torch.nn.functional as F
import numpy as np
from layers.rotaryembedding import RotaryEmbedding
from functools import partial
from tqdm import tqdm
import torch.nn as nn
from torch import nn, einsum
from torch.nn.modules import loss
from einops import rearrange, repeat
import math
from layers.samplers.dpm_sampler import DPMSolverSampler
from utils.diffusion_utils import *
from layers.RevIN import RevIN



def cosine_beta_schedule(timesteps, s=5):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)


class Model(nn.Module):
    
    def __init__(self, configs):
        super(Model, self).__init__()

        self.configs = configs
        self.device = 'cuda'
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.diff_steps = configs.diff_steps
        self.stride=configs.stride
        self.patch_len=configs.patch_len
        self.d_model=configs.d_model
        self.e_layers=configs.e_layers
        self.num_heads=configs.num_heads
        self.rmom_n = configs.rmom
        self.n_blocks = configs.n_b
        u_net = PatchUVIT(configs)
            
        self.enc_in = configs.enc_in
        self.batch_size =configs.batch_size
        self.beta_start = 1e-4 # 1e4
        self.beta_end = 1e-1#2e-2
        self.beta_schedule = 'cosine'
        self.v_posterior = 0.0
        self.loss_type = "l1"
        self.set_new_noise_schedule(None, self.beta_schedule, self.diff_steps, self.beta_start, self.beta_end)
        self.total_N = len(self.alphas_cumprod)
        self.T = 1.
        self.eps = 1e-5
        self.nn = u_net
        self.sampler = DPMSolverSampler(configs,self.nn, self.device,self.alphas_cumprod,self.betas.device)
        self.revin_layer = RevIN(self.enc_in, affine=True, subtract_last=False)


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, sample_times=5):
        if self.training:
            return self.forward_train(x_enc, x_mark_enc, x_dec, x_mark_dec,
                                             enc_self_mask, dec_self_mask, dec_enc_mask)
        else:
            return self.forward_val_test(x_enc, x_mark_enc, x_dec, x_mark_dec,
                                            enc_self_mask, dec_self_mask, dec_enc_mask, sample_times)


    def forward_train(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        #print(np.shape(x_enc),np.shape(x_mark_enc)) # (B,L,N)
        x = x_dec[:,-self.configs.pred_len:,:].permute(0,2,1) 
        f_dim = -1 if self.configs.features in ['MS'] else 0
        x=x[:,f_dim:,:]
        cond_ts = x_enc#(B,L,N)
        if self.configs.new_norm:
            cond_ts = self.revin_layer(cond_ts,'norm')
            cond_ts = cond_ts.permute(0,2,1) #(B,N,L)
            lenth = np.shape(x)[1] 
            mean_ = torch.mean(x[:,-lenth:,:], dim=1).unsqueeze(1)
            std_ = torch.ones_like(torch.std(x, dim=1).unsqueeze(1))
            x = (x-mean_.repeat(1,lenth,1))/(std_.repeat(1,lenth,1)+0.00001)
            B = np.shape(x)[0]
            N = self.configs.enc_in
            L1 = np.shape(cond_ts)[2]
            L2 = np.shape(x)[2]
            cond_ts = torch.reshape(cond_ts,(B*N,L1))
            x = torch.reshape(x,(B*N,L2))
            t = torch.randint(0, self.num_timesteps, size=[B*N//2,]).long().to(self.device)
            t = torch.cat([t, self.num_timesteps-1-t], dim=0)
            #print(t,t.shape)
            noise = torch.randn_like(x)
            x_k = self.noise_ts(x_start=x, t=t, noise=noise)
            model_out= self.nn(x_k, t, cond_ts,x_mark_enc)
            model_out=torch.reshape(model_out,(B,N,L2))
            model_out = model_out.permute(0,2,1) #(B,TARGET L,N)
            model_out=self.revin_layer(model_out,'denorm')
            model_out = model_out.permute(0,2,1)  #(B,N,TARGET L)
            weight_tmp = self.sqrt_one_minus_alphas_cumprod[t].reshape(model_out.shape[0],model_out.shape[1],1)
        else:
            cond_ts = cond_ts.permute(0,2,1) #(B,N,L) 
            mean_ = torch.mean(cond_ts, dim=-1,keepdims=True)
            std_ = torch.std(cond_ts, dim=-1,keepdims=True)
            cond_ts = (cond_ts-mean_)/(std_+0.00001)
            x = (x-mean_)/(std_+0.00001)
            B = np.shape(x)[0]
            N = self.configs.enc_in
            L1 = np.shape(cond_ts)[2]
            L2 = np.shape(x)[2]
            cond_ts = torch.reshape(cond_ts,(B*N,L1))
            x = torch.reshape(x,(B*N,L2))
            t = torch.randint(0, self.num_timesteps, size=[B*N//2,]).long().to(self.device)
            t = torch.cat([t, self.num_timesteps-1-t], dim=0)
            #print(t,t.shape)
            noise = torch.randn_like(x)
            x_k = self.noise_ts(x_start=x, t=t, noise=noise)
            model_out = self.nn(x_k, t, cond_ts,x_mark_enc)
            model_out = torch.reshape(model_out,(B,N,L2))
            model_out = model_out*(std_+0.00001) + mean_#(B,N,TARGET L)
            weight_tmp = self.sqrt_one_minus_alphas_cumprod[t].reshape(model_out.shape[0],model_out.shape[1],1)
        return model_out,weight_tmp

    def forward_val_test(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, sample_times=5):

        x_future = x_dec[:,-self.configs.pred_len:,:].permute(0,2,1)
        x_past = x_enc.permute(0,2,1)     
        f_dim = -1 if self.configs.features in ['MS'] else 0
        batchs, nF, nL = np.shape(x_past)[0], self.enc_in, self.pred_len
        batchs = batchs*self.enc_in
        if self.configs.features in ['MS']:
            nF = 1
        shape = [nF, nL]
        all_outs = []
        B = np.shape(x_past)[0]
        N = self.configs.enc_in
        if self.configs.new_norm:
            x_past = x_past.permute(0,2,1) #(B,L,N)
            x_past = self.revin_layer(x_past,'norm')
            x_past = x_past.permute(0,2,1) #(B,N,L)
        else:
            mean_ = torch.mean(x_past, dim=-1,keepdims=True)
            std_ = torch.std(x_past, dim=-1,keepdims=True)
            x_past = (x_past-mean_)/(std_+0.00001)
        x_past = torch.reshape(x_past,(B*N,-1))
        x_past=x_past.to(device='cuda:0')
        for i in range(sample_times):
            start_code = torch.randn((batchs, nL), device='cuda:0')
            diff_samples ,_= self.sampler.sample(S=self.configs.s_steps,
                                             conditioning=x_past,
                                             x_mark_enc=x_mark_enc,
                                             batch_size=batchs,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=1.0,
                                             unconditional_conditioning=None,
                                             eta=0.,
                                             x_T=start_code)
            diff_samples=torch.reshape(diff_samples,(B,N,-1))      
            if self.configs.new_norm:                       
                diff_samples = diff_samples.permute(0,2,1).to(self.device) #(B,TARGET L,N)
                diff_samples = self.revin_layer(diff_samples,'denorm')
            else:
                diff_samples = diff_samples.to(self.device)*(std_+0.00001) + mean_
                diff_samples = diff_samples.permute(0,2,1)
            all_outs.append(diff_samples)
        all_outs = torch.stack(all_outs, dim=0)
        if self.configs.use_mom and sample_times>1:
                outs = self._rob_median_of_means(all_outs)
        else:
                outs = all_outs.mean(0)
        
        return outs,all_outs.permute(1,0,2,3)

    def set_new_noise_schedule(self, given_betas=None, beta_schedule="linear", diff_steps=1000, beta_start=1e-4, beta_end=2e-2
    ):  

        betas = cosine_beta_schedule(diff_steps,self.configs.coss)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = beta_start
        self.linear_end = beta_end

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))
        lvlb_weights = 0.8 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))


        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all() 

    def noise_ts(self, x_start, t, noise=None):

        noise = default(noise, lambda: self.scaling_noise * torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)
    
    def _emp_mean(self, seq):
        return torch.sum(seq, dim=0) / seq.size(0)

    def _median_of_means(self, tensor):
        if self.n_blocks > tensor.size(0):
            self.n_blocks = int(torch.ceil(tensor.size(0) / 2))

        indic = torch.randperm(tensor.size(0))
        tensor = tensor[indic]  # Shuffle the tensor according to indic
        block_size = tensor.size(0) // self.n_blocks

        means = []
        for i in range(self.n_blocks):
            start_index = i * block_size
            end_index = start_index + block_size if (i+1) < self.n_blocks else tensor.size(0)
            block = tensor[start_index:end_index]
            block_mean = self._emp_mean(block)
            means.append(block_mean)

        means = torch.stack(means)
        return torch.median(means, dim=0)[0]

    def _rob_median_of_means(self, outputs):
        results = []
        for _ in range(self.rmom_n):
            shuffled_outputs = outputs[torch.randperm(outputs.size(0))]
            result = self._median_of_means(shuffled_outputs)
            results.append(result)
        results = torch.stack(results)
        return self._emp_mean(results)


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class PatchUVIT(nn.Module):
    def __init__(self,configs,**kwargs):    
        super().__init__()    
        self.model = FormerBone(configs)
        self.enc_in = configs.enc_in
        # self.task_name = config.task_name
    def forward(self, x, timesteps, cond_ts, x_mark_enc=None,*configs, **kwargs):    
        x = rearrange(x, '(b n) h  -> b n h', n = self.enc_in)
        cond_ts = rearrange(cond_ts, '(b n) h  -> b n h', n = self.enc_in)
        timesteps = rearrange(timesteps, '(b n)  -> b n', n = self.enc_in).unsqueeze(-1).unsqueeze(-1)     
        x= self.model(x, timesteps, cond_ts)
        return x
     
class FormerBone(nn.Module):
    def __init__(self, configs):    
        super().__init__()
        
        self.patch_len  =configs.patch_len
        self.stride = configs.stride
        self.d_model = configs.d_model
        self.n_blocks = configs.n_b
        patch_num = int((configs.seq_len - self.patch_len)/self.stride + 1)     
        
        patch_num_forecast = int((configs.pred_len - self.patch_len)/self.stride + 1)
        
        self.patch_num = patch_num
        self.patch_num_forecast = patch_num_forecast
        configs.enc_in = configs.enc_in
        configs.d_ff = configs.d_model * 2
             
        self.W_input_projection = nn.Linear(self.patch_len, configs.d_model)  
        self.input_dropout  = nn.Dropout(configs.dropout) 
        
        #self.cls = nn.Linear(1,configs.d_model)  
        self.cls = nn.Sequential(
            nn.Linear(1, configs.d_model),
            #nn.SiLU(),
        ) 
        
        self.W_outs = nn.Linear((patch_num+ 1+ patch_num_forecast)*configs.d_model, configs.pred_len) 
        self.Attentions_over_token = nn.ModuleList([Attenion(configs) for i in range(configs.e_layers)])
        self.Attentions_over_token_mid = Attenion(configs) 
        self.Attentions_over_token_up = nn.ModuleList([Attenion(configs) for i in range(configs.e_layers)])
        self.Attentions_mlp = nn.ModuleList([nn.Linear(configs.d_model*2,configs.d_model)  for i in range(configs.e_layers)])
        self.Attentions_dropout = nn.ModuleList([nn.Dropout(configs.skip_dropout)  for i in range(configs.e_layers)])
        self.Attentions_dropout_mid = nn.Dropout(configs.skip_dropout) 
        self.Attentions_dropout_up = nn.ModuleList([nn.Dropout(configs.skip_dropout)  for i in range(configs.e_layers)])
        self.Attentions_norm = nn.ModuleList([nn.Sequential(Transpose(1,2), nn.BatchNorm1d(configs.d_model), Transpose(1,2)) for i in range(configs.e_layers)])       
                

    def forward(self, x, timesteps, cond_ts,x_mark_enc=None):     
        b,c,s = x.shape
        zcube0 = cond_ts.unfold(dimension=-1, size=self.patch_len, step=self.stride)  
        zcube1 = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)           
        zcube = torch.cat([zcube0,zcube1],dim = -2)        
        z_embed = self.input_dropout(self.W_input_projection(zcube)) 
        time_token = self.cls(timesteps.float())
        z_embed = torch.cat((time_token,z_embed),dim = -2) 
        inputs = z_embed
        b,c,t,h = inputs.shape 
        skip = []
        for a_2,mlp,drop,norm  in zip(self.Attentions_over_token,self.Attentions_mlp ,self.Attentions_dropout,self.Attentions_norm ):
            output = a_2(inputs)
            inputs = drop(output)
            skip.append(inputs)
            
        inputs = self.Attentions_over_token_mid(inputs)
        inputs = self.Attentions_dropout_mid(inputs)
        for a_2,mlp,drop,norm  in zip(self.Attentions_over_token_up,self.Attentions_mlp ,self.Attentions_dropout,self.Attentions_norm ):
            prev = skip.pop()
            outputs = drop(mlp(torch.cat((prev,inputs),dim = -1))) 
            outputs = norm(outputs.reshape(b*c,t,-1)).reshape(b,c,t,-1) 
            output = a_2(inputs)
            inputs = drop(output)

        z_out = self.W_outs(output[:,:,:,:].reshape(b,c,-1)).reshape(b*c,-1)  
        return z_out
    

class Attenion(nn.Module):
    def __init__(self,config, over_hidden = False,trianable_smooth = False,untoken = False, *configs, **kwargs):
        super().__init__()

        self.over_hidden = over_hidden
        self.untoken = untoken
        self.num_heads = config.num_heads
        self.c_in = config.enc_in
        self.qkv = nn.Linear(config.d_model, config.d_model * 3, bias=True)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.head_dim = config.d_model // config.num_heads
        self.dropout_mlp = nn.Dropout(config.dropout)
        self.mlp = nn.Linear( config.d_model,  config.d_model)
        self.norm_post1  = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(config.d_model), Transpose(1,2))
        self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(config.d_model), Transpose(1,2))
        self.ff_1 = nn.Sequential(nn.Linear(config.d_model, config.d_ff, bias=True),
                        nn.GELU(),
                        nn.Dropout(config.dropout),
                        nn.Linear(config.d_ff, config.d_model, bias=True)
                       )
        self.rotary_emb = RotaryEmbedding(dim = self.head_dim//2)
    def forward(self, src, *configs,**kwargs):

        B,nvars, H, C, = src.shape
        qkv = self.qkv(src).reshape(B,nvars, H, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1,4, 2, 5)
        q, k, v = qkv[0].reshape(B*nvars,self.num_heads,-1,self.head_dim), qkv[1].reshape(B*nvars,self.num_heads,-1,self.head_dim), qkv[2].reshape(B*nvars,self.num_heads,-1,self.head_dim)
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        output1 = rearrange(x, '(b n) h e d -> b n  e (h d)',b = B)
        src2 =  self.ff_1(output1)
        src = src + src2
        src = src.reshape(B*nvars, -1, self.num_heads * self.head_dim)
        src = self.norm_attn(src)

        src = src.reshape(B,nvars, -1, self.num_heads * self.head_dim)
        return src
