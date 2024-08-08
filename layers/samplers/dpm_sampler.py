"""SAMPLING ONLY."""

import torch

from .dpm_solver import NoiseScheduleVP, model_wrapper, DPM_Solver


class DPMSolverSampler(object):
    def __init__(self, args,model, df_device,df_alphas_cumprod,df_betas_device, **kwargs):
        super().__init__()
        self.args=args
        self.diff_steps=args.diff_steps,
        self.lower_order_final = True if args.lower_order_final.lower() == 'true' else False

        self.model = model
        self.df_device=df_device
        self.df_alphas_cumprod=df_alphas_cumprod
        self.df_betas_device=df_betas_device
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.df_device)
        self.register_buffer('alphas_cumprod', to_torch(self.df_alphas_cumprod))
    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               x_mark_enc=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        # sampling
        F, L = shape
        size = (batch_size,  L)

        # print(f'Data shape for DPM-Solver sampling is {size}, sampling steps {S}')

        device = self.df_betas_device
        if x_T is None:
            img = torch.randn(size, device=device)
        else:
            img = x_T

        ns = NoiseScheduleVP('discrete', alphas_cumprod=self.alphas_cumprod)
        model_fn = model_wrapper(
                self.diff_steps,
                lambda x, t, c, m: self.model.forward(x, t, c, m),
                ns,
                model_type="x_start",
                guidance_type="classifier-free",
                condition=conditioning,
                x_mark_enc=x_mark_enc,
                unconditional_condition=unconditional_conditioning,
                guidance_scale=unconditional_guidance_scale,
            )
        dpm_solver = DPM_Solver(model_fn, ns, predict_x0=True, thresholding=False)
        x = dpm_solver.sample(img, steps=S, skip_type=self.args.skip_type, method=self.args.method, order=self.args.order, lower_order_final=self.lower_order_final)

        return x.to(device), None

    '''
                - 'logSNR': uniform logSNR for the time steps.
                - 'time_uniform': uniform time for the time steps. (**Recommended for high-resolutional data**.)
                - 'time_quadratic': quadratic time for the time steps. (Used in DDIM for low-resolutional data.)
    '''

        