import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def extract(v, t, x_shape):
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

def total_variation(log_var: torch.Tensor) -> torch.Tensor:
    dh = log_var[:, :, 1:, :] - log_var[:, :, :-1, :]
    dw = log_var[:, :, :, 1:] - log_var[:, :, :, :-1]
    return (dh.abs().mean() + dw.abs().mean())
    

class CE_DDIM_trainer(nn.Module):
     
    def __init__(self, model, beta_1, beta_T, T, num_channel, noise_scheduler):
        super().__init__()
        self.model = model
        self.T = T
        self.num_channel = num_channel
        self.noise_scheduler = noise_scheduler
        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).float())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar).float())
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar).float())
        self.LAM_NLL_BASE = 1.0
        self.N_WARM = 100
    
    def forward(self, x_0, lam_nll_param):
        B = x_0.size(0)
        t = torch.randint(0, self.T, (B,), device=x_0.device, dtype=torch.long)

        ct   = x_0[:, :self.num_channel]       # ground-truth CT
        cbct = x_0[:, self.num_channel:]        # conditioning CBCT
        # ───────────────── forward diffusion ────────────────────
        noise = torch.randn_like(ct)
        x_t   = (extract(self.sqrt_alphas_bar, t, ct.shape) * ct +
                 extract(self.sqrt_one_minus_alphas_bar, t, ct.shape) * noise)
        eps_true = noise                        # ground-truth noise
        # ───────────────── model prediction ─────────────────────
        eps_hat, log_var_hat = self.model(torch.cat([x_t, cbct], dim=1), t)
        log_var_hat = torch.clamp(log_var_hat, min=-14.0, max=2.0)
        w_t = 1. / (extract(self.sqrt_one_minus_alphas_bar, t, ct.shape) ** 2)

        # ───────────────── loss computation ─────────────────────
        # Compute prediction error for variance loss
        diff_sq = (eps_true - eps_hat).pow(2)
        L_eps = (eps_hat, eps_true).pow(2).detach() # For monitoring only

        # L_var: Negative log-likelihood loss
        var_inv = torch.exp(-log_var_hat)      # 1/σ̂²
        L_var_pixel = w_t * (var_inv * diff_sq + log_var_hat)
        L_var = L_var_pixel.mean() / w_t.mean()

        LOG_SIGMA0_SQ = 0.0     # Expected noise variance (ε ~ N(0,1))
        LAMBDA_TV     = 1e-6    # Total variation weight
        LAMBDA_PRIOR  = 1e-6    # Prior regularization weight

        tv_loss = LAMBDA_TV * total_variation(log_var_hat.float())
        prior_loss = LAMBDA_PRIOR * (log_var_hat - LOG_SIGMA0_SQ).pow(2).mean()

        return L_eps, L_var, tv_loss, prior_loss


class CE_DDIM_sampler(nn.Module):

    def __init__(self, model, sample_steps=150,
                 skip_steps=1, eta=0.0,
                 num_channel=3, alpha_star=1.0,
                 hu_range=(-1000, 3000),
                 sigma_thresh_hu=20):
        super().__init__()
        self.model = model        
        self.device = next(model.parameters()).device
        self.T = 1000
        self.sample_steps = sample_steps
        self.skip_steps   = skip_steps
        self.eta          = eta
        self.nc           = num_channel
        self.alpha_star   = alpha_star
        self.sigma_thresh = sigma_thresh_hu
        self.beta_1 = 4e-4
        self.beta_T = 0.02
        self.refine = False
        
        betas = torch.linspace(self.beta_1, self.beta_T, self.T, dtype=torch.float32)
        betas = betas.to(self.device, dtype=torch.float32)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', 1 - betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(1 - betas, dim=0))

        self.HU_min, self.HU_max = hu_range
        self.hu_scale = self.HU_max - self.HU_min  

    # ---------- helpers --------------------------------------------------
    def _extract(self, buf, t, shape):
        out = buf.gather(0, t).float()
        while out.ndim < len(shape): out = out[..., None]
        return out

    @torch.no_grad()

    def reconstruct(self, x_T: torch.Tensor,
                    seed_noise: torch.Tensor = None,
                    refine: bool = False):

        def safe_sqrt(x, eps=1e-12):
            return torch.sqrt(torch.clamp(x, min=eps))

        B, _, H, W = x_T.shape
        device = self.device
        x_t = x_T[:, :self.nc, :, :] 
        cbct = x_T[:, self.nc:, :, :]  
        ts = np.linspace(self.T - 1, 0, self.sample_steps).round().astype(int)
        ts = ts[::self.skip_steps]
        if ts[-1] != 0:
            ts = np.append(ts, 0)

        mu_out    = None
        sigma_out = None

        for idx in range(len(ts) - 1):
            t_cur  = torch.full((B,), ts[idx],   device=device, dtype=torch.long)

            t_next = torch.full((B,), ts[idx+1], device=device, dtype=torch.long)
            # --- predict ε̂ and log σ̂²_ε -----------------------------------
            eps_hat, log_var_hat = self.model(torch.cat([x_t, cbct], dim=1), t_cur)
            log_var_hat = torch.clamp(log_var_hat, -12.0, 3.0)
            sigma_eps2  = torch.exp(log_var_hat)
            # --- coefficients -------------------------------------------------
            alpha_t     = self._extract(self.alphas_cumprod, t_cur,  x_t.shape)
            alpha_prev  = self._extract(self.alphas_cumprod, t_next, x_t.shape)
            beta_t      = 1.0 - alpha_t
            # --- x0 mean/var (analytic reparam) ------------------------------
            mu_x0    = (x_t - beta_t.sqrt() * eps_hat) / safe_sqrt(alpha_t)
            mu_x0    = torch.clamp(mu_x0, 0.0, 1.0)  # stay in normalized range
            sigma_x0 = safe_sqrt((1 - alpha_t) / alpha_t) * safe_sqrt(sigma_eps2)
            # store final step outputs
            if idx == len(ts) - 2:
                mu_out    = mu_x0
                sigma_out = sigma_x0 * self.alpha_star  # conformal scaling later
            # --- DDIM update --------------------------------------------------
            frac = ((1 - alpha_prev) / (1 - alpha_t)) * (1 - alpha_t / alpha_prev)
            sigma_t = self.eta * safe_sqrt(frac)

            # sqrt term must be positive
            dir_scale = safe_sqrt(torch.clamp(1 - alpha_prev - sigma_t**2, min=0.0))

            noise = sigma_t * torch.randn_like(x_t) if self.eta > 0 else 0.0
            x_t   = safe_sqrt(alpha_prev) * mu_x0 + dir_scale * eps_hat + noise

        mu_hu    = (mu_out * 0.5 + 0.5) * self.hu_scale + self.HU_min
        sigma_hu = sigma_out * self.hu_scale

        return mu_out, sigma_out, mu_hu, sigma_hu