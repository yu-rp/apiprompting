import time
import numpy as np
import torch
from PIL import Image
import glob
import sys
import argparse
import datetime
import json
from pathlib import Path

class MaskHookLogger(object):
    def __init__(self, model, device):
        self.current_layer = 0
        self.device = device
        self.attentions = []
        self.mlps = []
        self.post_ln_std = None
        self.post_ln_mean = None
        self.model = model
        
    @torch.no_grad()
    def compute_attentions(self, ret):
        if self.current_layer == self.layer_index:
            bias_term = self.model.visual.transformer.resblocks[self.current_layer].attn.out_proj.bias
            return_value = ret[:, 0]
            return_value = return_value +  bias_term[np.newaxis, np.newaxis] / (return_value.shape[1])# [b, n, d]
            self.attentions.append(return_value.detach()) 
        self.current_layer += 1
        return ret
    
    @torch.no_grad()
    def compute_mlps(self, ret):
        if self.current_layer == self.layer_index + 1:
            self.mlps.append(ret[:, 1:].detach()) # [b, n, d] 
        return ret
    
    @torch.no_grad()
    def log_post_ln_mean(self, ret):
        self.post_ln_mean = ret.detach() # [b, 1]
        return ret
        
    @torch.no_grad()
    def log_post_ln_std(self, ret):
        self.post_ln_std = ret.detach() # [b, 1] 
        return ret
    
    def _normalize_mlps(self):
        len_intermediates = self.current_layer * 2 - 1
        # This is just the normalization layer:
        mean_centered = (self.mlps - 
                         self.post_ln_mean[:, :, np.newaxis, np.newaxis] / len_intermediates)

        weighted_mean_centered = self.model.visual.ln_post.weight.detach() * mean_centered
        weighted_mean_by_std = weighted_mean_centered / self.post_ln_std[:, :, np.newaxis, np.newaxis]

        bias_term = self.model.visual.ln_post.bias.detach() / len_intermediates
        post_ln = weighted_mean_by_std + bias_term
        return post_ln @ self.model.visual.proj.detach()
    
    def _normalize_attentions(self):
        len_intermediates = self.current_layer * 2 - 1 # 2*l + 1
        normalization_term = self.attentions.shape[2] * 1  # n * h, h=1
        # This is just the normalization layer:
        mean_centered = (self.attentions - 
                         self.post_ln_mean[:, :, np.newaxis, np.newaxis] / 
                         (len_intermediates * normalization_term))
        weighted_mean_centered = self.model.visual.ln_post.weight.detach() * mean_centered
        weighted_mean_by_std = weighted_mean_centered / self.post_ln_std[:, :, np.newaxis, np.newaxis]
        bias_term = self.model.visual.ln_post.bias.detach() / (len_intermediates * normalization_term)
        post_ln = weighted_mean_by_std + bias_term
        return post_ln @ self.model.visual.proj.detach()
    
    @torch.no_grad()
    def finalize(self, representation):
        """We calculate the post-ln scaling, project it and normalize by the last norm."""
        self.attentions = torch.stack(self.attentions, axis=1) # [b, 1, n, d]
        self.mlps = torch.stack(self.mlps, axis=1) # [b, 1, n, d]
        projected_attentions = self._normalize_attentions()
        projected_mlps = self._normalize_mlps()
        norm = representation.norm(dim=-1).detach()
        return (projected_attentions / norm[:, np.newaxis, np.newaxis, np.newaxis], 
                projected_mlps / norm[:, np.newaxis, np.newaxis, np.newaxis])
        
    def reinit(self):
        self.current_layer = 0
        self.attentions = []
        self.mlps = []
        self.post_ln_mean = None
        self.post_ln_std = None
        torch.cuda.empty_cache()


def hook_prs_logger(model, device, layer_index = 23):
    """Hooks a projected residual stream logger to the model."""
    prs = MaskHookLogger(model, device)
    model.hook_manager.register('visual.transformer.resblocks.*.attn.out.post', 
                                prs.compute_attentions)

    model.hook_manager.register('visual.transformer.resblocks.*.post', 
                                prs.compute_mlps)
    model.hook_manager.register('visual.ln_pre_post', 
                                prs.compute_mlps)
    model.hook_manager.register('visual.ln_post.mean', 
                                prs.log_post_ln_mean)
    model.hook_manager.register('visual.ln_post.sqrt_var', 
                                prs.log_post_ln_std)
    prs.layer_index = layer_index

    return prs
