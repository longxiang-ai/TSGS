# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.quaternion_utils import init_predefined_omega

# Helper function to create positional encoding embedder
def get_embedder(multires, i=1):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': i,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


# Positional encoding implementation (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


# Positional encoding helper function
def positional_encoding(positions, freqs):
    freq_bands = (2 ** torch.arange(freqs).float()).to(positions.device)  # (F,)
    pts = (positions[..., None] * freq_bands).reshape(
        positions.shape[:-1] + (freqs * positions.shape[-1],))  # (..., DF)
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts


# Module for rendering equation encoding using spherical gaussians
class RenderingEquationEncoding(torch.nn.Module):
    def __init__(self, num_theta, num_phi, device):
        super(RenderingEquationEncoding, self).__init__()

        self.num_theta = num_theta
        self.num_phi = num_phi

        # Initialize predefined spherical coordinates
        omega, omega_la, omega_mu = init_predefined_omega(num_theta, num_phi)
        self.omega = omega.view(1, num_theta, num_phi, 3).to(device)
        self.omega_la = omega_la.view(1, num_theta, num_phi, 3).to(device)
        self.omega_mu = omega_mu.view(1, num_theta, num_phi, 3).to(device)

    def forward(self, omega_o, a, la, mu):
        # Calculate smooth term
        Smooth = F.relu((omega_o[:, None, None] * self.omega).sum(dim=-1, keepdim=True))  # N, num_theta, num_phi, 1

        # Apply softplus activation to lambda and mu
        la = F.softplus(la - 1)
        mu = F.softplus(mu - 1)
        
        # Calculate exponential input
        exp_input = -la * (self.omega_la * omega_o[:, None, None]).sum(dim=-1, keepdim=True).pow(2) - mu * (
                self.omega_mu * omega_o[:, None, None]).sum(dim=-1, keepdim=True).pow(2)
        out = a * Smooth * torch.exp(exp_input)
        # ASG(v) = a * max(v·ω, 0) * exp(-λ_1(v·ω_λ)^2 - λ_2(v·ω_μ)^2)
        return out


# Environment map represented using spherical gaussians
class SGEnvmap(torch.nn.Module):
    def __init__(self, numLgtSGs=32, device='cuda'):
        super(SGEnvmap, self).__init__()

        # Initialize SG parameters
        self.lgtSGs = nn.Parameter(torch.randn(numLgtSGs, 7).cuda())  # lobe + lambda + mu
        self.lgtSGs.data[..., 3:4] *= 100.
        self.lgtSGs.data[..., -3:] = 0.
        self.lgtSGs.requires_grad = True

    def forward(self, viewdirs):
        # Extract and normalize SG parameters
        lgtSGLobes = self.lgtSGs[..., :3] / (torch.norm(self.lgtSGs[..., :3], dim=-1, keepdim=True) + 1e-7)
        lgtSGLambdas = torch.abs(self.lgtSGs[..., 3:4])  # sharpness
        lgtSGMus = torch.abs(self.lgtSGs[..., -3:])  # positive values
        
        # Calculate radiance
        pred_radiance = lgtSGMus[None] * torch.exp(
            lgtSGLambdas[None] * (torch.sum(viewdirs[:, None, :] * lgtSGLobes[None], dim=-1, keepdim=True) - 1.))
        reflection = torch.sum(pred_radiance, dim=1)

        return reflection


# Anisotropic Spherical Gaussian Renderer
class ASGRender(torch.nn.Module):
    def __init__(self, viewpe=2, featureC=128, num_theta=4, num_phi=8):
        super(ASGRender, self).__init__()

        # Initialize parameters
        self.num_theta = num_theta
        self.num_phi = num_phi
        self.ch_normal_dot_viewdir = 1
        self.in_mlpC = 2 * viewpe * 3 + 3 + self.num_theta * self.num_phi * 2 + self.ch_normal_dot_viewdir
        self.viewpe = viewpe
        self.ree_function = RenderingEquationEncoding(self.num_theta, self.num_phi, 'cuda')

        # Build MLP network
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, 3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def reflect(self, viewdir, normal):
        # Calculate reflection direction
        out = 2 * (viewdir * normal).sum(dim=-1, keepdim=True) * normal - viewdir
        return out

    def safe_normalize(self, x, eps=1e-8):
        # Normalize vector with epsilon for numerical stability
        return x / (torch.norm(x, dim=-1, keepdim=True) + eps)

    def forward(self, pts, viewdirs, features, normal):
        # Reshape ASG parameters
        asg_params = features.view(-1, self.num_theta, self.num_phi, 4)  # [N, 8, 16, 4]
        a, la, mu = torch.split(asg_params, [2, 1, 1], dim=-1)

        # Calculate reflection direction and color features
        reflect_dir = self.safe_normalize(self.reflect(-viewdirs, normal))
        color_feature = self.ree_function(reflect_dir, a, la, mu)
        color_feature = color_feature.view(color_feature.size(0), -1)  # [N, 256]

        # Calculate dot product and prepare input data
        normal_dot_viewdir = ((-viewdirs) * normal).sum(dim=-1, keepdim=True)  # [N, 1]
        indata = [color_feature, normal_dot_viewdir]
        if self.viewpe > -1:
            indata += [viewdirs]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)

        return rgb


# ASG Renderer for real scenes
class ASGRenderReal(torch.nn.Module):
    def __init__(self, viewpe=2, featureC=32, num_theta=2, num_phi=4, is_indoor=False):
        super(ASGRenderReal, self).__init__()

        # Initialize parameters
        self.num_theta = num_theta
        self.num_phi = num_phi
        self.in_mlpC = 2 * viewpe * 3 + 3 + self.num_theta * self.num_phi * 2
        self.viewpe = viewpe
        self.ree_function = RenderingEquationEncoding(self.num_theta, self.num_phi, 'cuda')

        # Build MLP network
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, 3)

        # Different network structure for indoor/outdoor scenes
        if is_indoor:
            self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        else:
            self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer3)

        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features, normal):
        # Process features and calculate color
        asg_params = features.view(-1, self.num_theta, self.num_phi, 4)  # [N, 8, 16, 4]
        a, la, mu = torch.split(asg_params, [2, 1, 1], dim=-1)

        color_feature = self.ree_function(viewdirs, a, la, mu)
        color_feature = color_feature.view(color_feature.size(0), -1)  # [N, 256]

        # Prepare input data
        indata = [color_feature]
        if self.viewpe > -1:
            indata += [viewdirs]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)

        return rgb


# Network for specular component
class SpecularNetwork(nn.Module):
    def __init__(self):
        super(SpecularNetwork, self).__init__()

        # Initialize network parameters
        self.asg_feature = 24
        self.num_theta = 4
        self.num_phi = 8
        self.view_pe = 2
        self.hidden_feature = 128
        self.asg_hidden = self.num_theta * self.num_phi * 4

        # Define network layers
        self.gaussian_feature = nn.Linear(self.asg_feature, self.asg_hidden)
        self.render_module = ASGRender(self.view_pe, self.hidden_feature, self.num_theta, self.num_phi)

    def forward(self, x, view, normal):
        # Process features and render specular component
        feature = self.gaussian_feature(x)
        spec = self.render_module(x, view, feature, normal)
        return spec


# Network for specular component in real scenes
class SpecularNetworkReal(nn.Module):
    def __init__(self, is_indoor=False):
        super(SpecularNetworkReal, self).__init__()

        # Initialize network parameters
        self.asg_feature = 12
        self.num_theta = 2
        self.num_phi = 4
        self.view_pe = 2
        self.hidden_feature = 32
        self.asg_hidden = self.num_theta * self.num_phi * 4

        # Define network layers
        self.gaussian_feature = nn.Linear(self.asg_feature, self.asg_hidden)
        self.render_module = ASGRenderReal(self.view_pe, self.hidden_feature, self.num_theta, self.num_phi, is_indoor)

    def forward(self, x, view, normal):
        # Process features and render specular component
        feature = self.gaussian_feature(x)
        spec = self.render_module(x, view, feature, normal)
        return spec


# Network for anchor-based specular component
class AnchorSpecularNetwork(nn.Module):
    def __init__(self, feature_dims):
        super(AnchorSpecularNetwork, self).__init__()

        # Initialize network parameters
        self.asg_feature = feature_dims
        self.num_theta = 2
        self.num_phi = 4
        self.asg_hidden = self.num_theta * self.num_phi * 4

        # Define network layers
        self.gaussian_feature = nn.Linear(self.asg_feature + 3, self.asg_hidden)
        self.gaussian_diffuse = nn.Linear(self.asg_feature, 3)
        self.gaussian_normal = nn.Linear(self.asg_feature + 3, 3)
        self.render_module = ASGRender(2, 64, num_theta=2, num_phi=4)

    def forward(self, x, view, normal_center, offset):
        # Process features and combine diffuse and specular components
        feature = self.gaussian_feature(torch.cat([x, view], dim=-1))
        diffuse = self.gaussian_diffuse(x)
        normal_delta = self.gaussian_normal(torch.cat([x, offset], dim=-1))
        normal = F.normalize(normal_center + normal_delta, dim=-1)
        spec = self.render_module(x, view, feature, normal)
        rgb = diffuse + spec

        return rgb
