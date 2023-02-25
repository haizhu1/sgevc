import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

import commons
import modules
import attentions
import monotonic_align

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from commons import init_weights, get_padding
from layers import *
import numpy as np

class StochasticDurationPredictor(nn.Module):
  def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, n_flows=4, gin_channels=0):
    super().__init__()
    filter_channels = in_channels # it needs to be removed from future version.
    self.in_channels = in_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.n_flows = n_flows
    self.gin_channels = gin_channels

    self.log_flow = modules.Log()
    self.flows = nn.ModuleList()
    self.flows.append(modules.ElementwiseAffine(2))
    for i in range(n_flows):
      self.flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
      self.flows.append(modules.Flip())

    self.post_pre = nn.Conv1d(1, filter_channels, 1)
    self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
    self.post_convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
    self.post_flows = nn.ModuleList()
    self.post_flows.append(modules.ElementwiseAffine(2))
    for i in range(4):
      self.post_flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
      self.post_flows.append(modules.Flip())

    self.pre = nn.Conv1d(in_channels, filter_channels, 1)
    self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
    self.convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
    if gin_channels != 0:
      self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

  def forward(self, x, x_mask, w=None, g=None, reverse=False, noise_scale=1.0):
    x = torch.detach(x)
    x = self.pre(x)
    if g is not None:
      g = torch.detach(g)
      x = x + self.cond(g)
    x = self.convs(x, x_mask)
    x = self.proj(x) * x_mask

    if not reverse:
      flows = self.flows
      assert w is not None

      logdet_tot_q = 0 
      h_w = self.post_pre(w)
      h_w = self.post_convs(h_w, x_mask)
      h_w = self.post_proj(h_w) * x_mask
      e_q = torch.randn(w.size(0), 2, w.size(2)).to(device=x.device, dtype=x.dtype) * x_mask
      z_q = e_q
      for flow in self.post_flows:
        z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
        logdet_tot_q += logdet_q
      z_u, z1 = torch.split(z_q, [1, 1], 1) 
      u = torch.sigmoid(z_u) * x_mask
      z0 = (w - u) * x_mask
      logdet_tot_q += torch.sum((F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1,2])
      logq = torch.sum(-0.5 * (math.log(2*math.pi) + (e_q**2)) * x_mask, [1,2]) - logdet_tot_q

      logdet_tot = 0
      z0, logdet = self.log_flow(z0, x_mask)
      logdet_tot += logdet
      z = torch.cat([z0, z1], 1)
      for flow in flows:
        z, logdet = flow(z, x_mask, g=x, reverse=reverse)
        logdet_tot = logdet_tot + logdet
      nll = torch.sum(0.5 * (math.log(2*math.pi) + (z**2)) * x_mask, [1,2]) - logdet_tot
      return nll + logq # [b]
    else:
      flows = list(reversed(self.flows))
      flows = flows[:-2] + [flows[-1]] # remove a useless vflow
      z = torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype) * noise_scale
      for flow in flows:
        z = flow(z, x_mask, g=x, reverse=reverse)
      z0, z1 = torch.split(z, [1, 1], 1)
      logw = z0
      return logw


class DurationPredictor(nn.Module):
  def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0):
    super().__init__()

    self.in_channels = in_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.gin_channels = gin_channels

    self.drop = nn.Dropout(p_dropout)
    self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size//2)
    self.norm_1 = modules.LayerNorm(filter_channels)
    self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size//2)
    self.norm_2 = modules.LayerNorm(filter_channels)
    self.proj = nn.Conv1d(filter_channels, 1, 1)

    if gin_channels != 0:
      self.cond = nn.Conv1d(gin_channels, in_channels, 1)

  def forward(self, x, x_mask, g=None):
    x = torch.detach(x)
    if g is not None:
      g = torch.detach(g)
      x = x + self.cond(g)
    x = self.conv_1(x * x_mask)
    x = torch.relu(x)
    x = self.norm_1(x)
    x = self.drop(x)
    x = self.conv_2(x * x_mask)
    x = torch.relu(x)
    x = self.norm_2(x)
    x = self.drop(x)
    x = self.proj(x * x_mask)
    return x * x_mask

class Projection(nn.Module):
    def __init__(self, out_channels, hidden_channels):
        super().__init__()
        self.out_channels = out_channels
        self.proj= nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_mask):
        #eu_expand = eu.unsqueeze(2) # e:[b,h,1]
        #eu_repeat = eu_expand.repeat(1,1,x.shape[2])
        #add = x + eu_repeat
        #concat = torch.cat((add,ep),dim=1)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return m, logs

class TextEncoder(nn.Module):
    def __init__(self,
        n_vocab,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout):
        super().__init__()
        self.n_vocab = n_vocab
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        self.encoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)

    def forward(self, x, x_lengths):
        x = self.emb(x) * math.sqrt(self.hidden_channels) # [b, t, h]
        x = torch.transpose(x, 1, -1) # [b, h, t] [64,192,97]
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.encoder(x * x_mask, x_mask) # [b,h,t]
        return x, x_mask


class ResidualCouplingBlock(nn.Module):
  def __init__(self,
      channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      n_flows=4,
      gin_channels=0):
    super().__init__()
    self.channels = channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.n_flows = n_flows
    self.gin_channels = gin_channels

    self.flows = nn.ModuleList()
    for i in range(n_flows):
      self.flows.append(modules.ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True))
      self.flows.append(modules.Flip())

  def forward(self, x, x_mask, g=None, reverse=False):
    if not reverse:
      for flow in self.flows:
        x, _ = flow(x, x_mask, g=g, reverse=reverse)
    else:
      for flow in reversed(self.flows):
        x = flow(x, x_mask, g=g, reverse=reverse)
    return x

class PhonemeLevelPredictor(nn.Module):
    def __init__(self,
        hidden_channels):
        super().__init__()
        self.in_channels = hidden_channels
        self.hidden_channels = hidden_channels

        self.p_conv_layers = []
        in_filters = [hidden_channels, hidden_channels]
        out_filters = [hidden_channels, hidden_channels]
        for i in range(2):
            self.p_conv_layers.append(
                nn.Conv1d(
                    in_channels=in_filters[i],
                    out_channels=out_filters[i],
                    kernel_size=3,
                    stride=1,
                    padding=1
                    )
                )
            self.p_conv_layers.append(nn.ReLU())
            self.p_conv_layers.append(modules.LayerNorm(hidden_channels))
            self.p_conv_layers.append(torch.nn.Dropout(p=0.1))
        self.p_conv_layers_sequence = torch.nn.ModuleList(self.p_conv_layers)
        self.linear = torch.nn.Linear(hidden_channels, 4)
    def forward(self, x, x_mask):
        x = torch.detach(x)
        for i,layer in enumerate(self.p_conv_layers_sequence):
            x = layer(x*x_mask)
        x_transpose = x.permute(0,2,1)
        p_level = self.linear(x_transpose)
        p_level = p_level.permute(0,2,1)
        return p_level*x_mask

class PhonemeLevelEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        in_filters = [hidden_channels, hidden_channels]
        out_filters = [hidden_channels, hidden_channels]
        self.p_conv_layers = []
        self.pre = nn.Conv1d(in_channels,hidden_channels,1)
        for i in range(2):
            self.p_conv_layers.append(
                nn.Conv1d(
                    in_channels=in_filters[i],
                    out_channels=out_filters[i],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    )
                )
            self.p_conv_layers.append(nn.ReLU())
            self.p_conv_layers.append(modules.LayerNorm(hidden_channels))
            self.p_conv_layers.append(torch.nn.Dropout(p=0.1))
        self.p_conv_layers_sequence = torch.nn.ModuleList(self.p_conv_layers)
        self.linear = torch.nn.Linear(hidden_channels, 4)

    def forward(self, x, x_mask, w):
        # x: spec: torch.Size([2, 513, 278])
        # w: torch.Size([2, 83])
        # spec_split: [2,513,83]
        w = torch.detach(w.int())
        w = w.reshape([w.shape[0],-1])
        spec_split = torch.zeros([x.shape[0],x.shape[1],w.shape[-1]],dtype=x.dtype).cuda()
        for i in range(w.shape[0]):
            start=torch.tensor(0, dtype=w.dtype).cuda()
            for j in range(w.shape[-1]):
                if w[i][j]>=2:
                    end = start + w[i][j]
                    spec_split[i,:,j] = x[i,:,start:end].mean(dim=-1)
                elif w[i][j]==1:
                    end = start + 1
                    spec_split[i,:,j] = x[i,:,start:end].squeeze(-1)
                else:
                    spec_split[i,:,j] = torch.zeros(x.shape[1],dtype=x.dtype)
                start = end
        x = self.pre(spec_split)
        for i,layer in enumerate(self.p_conv_layers_sequence):
            x = layer(x*x_mask)
        x_transpose = x.permute(0,2,1)
        p_level = self.linear(x_transpose)
        p_level = p_level.permute(0,2,1)
        p_level = p_level*x_mask
        return p_level

# Inference networks Include PosteriorEncoder q(z|x,e) and q(e|x)
class PosteriorEncoder(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        hidden_channels,
        num_class,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_class = num_class
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.u_conv_layers = []
        in_filters = [in_channels, 256]
        out_filters = [256, 256]
        for i in range(2):
            self.u_conv_layers.append(
                nn.Conv1d(
                    in_channels=in_filters[i], 
                    out_channels=out_filters[i],
                    kernel_size=5,
                    stride=3,
                    padding=get_padding(5)
                    )
                )
            self.u_conv_layers.append(nn.ReLU())
            self.u_conv_layers.append(modules.LayerNorm(256))
            self.u_conv_layers.append(torch.nn.Dropout(p=0.1))

        self.u_conv_layers_sequence = torch.nn.ModuleList(self.u_conv_layers)
        self.gumbel_softmax = GumbelSoftmax(256, num_class, 32)#batch_size:64 num_class:10 num_class_dim: 16
        #self.gaussian = Gaussian(num_class, 256)
        self.linear = nn.Linear(num_class*32, 256)
        #self.softmax = nn.Softmax(dim=-1)
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def qex1(self, x, x_mask, temp): # Attribute Encoder
        # x: [b,h,t]
        x = x*x_mask
        for i,layer in enumerate(self.u_conv_layers_sequence):
            x = layer(x)
        x = torch.mean(x, dim=-1) #[b,h]
        e, log_q_e, q_e = self.gumbel_softmax(x, temperature=temp, hard=False) #y_, log_q_y, q_y: [b*N,K]   y: [b,N*K]
        e = e.reshape([x.shape[0],-1])
        eu = self.linear(e)
        #mu, var2, z = self.gaussian(e)
        return eu, e, log_q_e, q_e

    def qzxe(self, x, x_mask, g): #VC Encoder
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g)
        stats = self.proj(x) * x_mask #stats:[64,384,400]
    
        m, logs = torch.split(stats, self.out_channels, dim=1) #m,logs:[64, 192, 400]
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask #z:[64, 192, 400]
        return z, m, logs 

    def forward(self, x, x_lengths, g, temp, x2=None, x2_lengths=None, infer=False):
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        eu, e, log_q_e, q_e = self.qex1(x, x_mask, temp) # inference q(e|x)  eu:[b,h]
        eu = eu.unsqueeze(-1)

        if not infer:
            g = g+eu
            z, m, logs = self.qzxe(x, x_mask, g)# inference q(z|x,e) m,logs,z:[64, 192, 400]
            return z, m, logs, eu, e, log_q_e, q_e, x_mask
        else:
            if x2 is not None:
                x2_mask = torch.unsqueeze(commons.sequence_mask(x2_lengths, x2.size(2)), 1).to(x2.dtype)
                eu2, e2, log_q_e2, q_e2 = self.qex1(x2, x2_mask, temp)
                #print("eu2:",eu2.squeeze())
                eu2 = eu2.unsqueeze(-1)
                eu = 0.0*eu + 1.0*eu2
            return eu, e, log_q_e, q_e, x_mask
 
class Generator(torch.nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        #self.pre = Conv1d(initial_channel, initial_channel, 1, 1)
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock = modules.ResBlock1 if resblock == '1' else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        #x = self.pre(x)
        x = self.conv_pre(x)
        if g is not None:
          x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2,3,5,7,11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs



class SynthesizerTrn(nn.Module):
  """
  Synthesizer for Training
  """

  def __init__(self, 
    n_vocab,
    spec_channels,
    segment_size,
    inter_channels,
    hidden_channels,
    filter_channels,
    num_class,
    n_heads,
    n_layers,
    kernel_size,
    p_dropout,
    resblock, 
    resblock_kernel_sizes, 
    resblock_dilation_sizes, 
    upsample_rates, 
    upsample_initial_channel, 
    upsample_kernel_sizes,
    n_speakers=20,
    gin_channels=0,
    use_sdp=True,
    **kwargs):

    super().__init__()
    self.n_vocab = n_vocab
    self.spec_channels = spec_channels
    self.inter_channels = inter_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.resblock = resblock
    self.resblock_kernel_sizes = resblock_kernel_sizes
    self.resblock_dilation_sizes = resblock_dilation_sizes
    self.upsample_rates = upsample_rates
    self.upsample_initial_channel = upsample_initial_channel
    self.upsample_kernel_sizes = upsample_kernel_sizes
    self.segment_size = segment_size
    self.gin_channels = gin_channels
    self.n_speakers = n_speakers
    self.use_sdp = use_sdp
    self.enc_p = TextEncoder(n_vocab,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout)
    self.proj_p = Projection(inter_channels,hidden_channels)
    self.dec = Generator(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels)
    self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, num_class, 5, 1, 16, gin_channels=gin_channels)
    self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels)
    
    if use_sdp:
      self.dp = StochasticDurationPredictor(hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels)
    else:
      self.dp = DurationPredictor(hidden_channels, 256, 3, 0.5, gin_channels=gin_channels)

    self.emb_gs = nn.Embedding(n_speakers, gin_channels)

  def forward(self, x, x_lengths, y, y_lengths, sid, temp):
    """
        x:[64,97] 
        y:[64,513,400]
    """
    gs = self.emb_gs(sid).unsqueeze(-1) # [b, h, 1]

    x, x_mask = self.enc_p(x, x_lengths)
    m_p, logs_p = self.proj_p(x, x_mask)
    z, m_q, logs_q, eu, e, log_q_e, q_e, y_mask = self.enc_q(y, y_lengths, gs, temp)
    g = gs+eu
    z_p = self.flow(z, y_mask, g=g)

    with torch.no_grad():
      # negative cross-entropy
      s_p_sq_r = torch.exp(-2 * logs_p) # [b, d, t]
      neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True) # [b, 1, t_s]
      neg_cent2 = torch.matmul(-0.5 * (z_p ** 2).transpose(1, 2), s_p_sq_r) # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
      neg_cent3 = torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r)) # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
      neg_cent4 = torch.sum(-0.5 * (m_p ** 2) * s_p_sq_r, [1], keepdim=True) # [b, 1, t_s]
      neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4
      attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
      attn = monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()
    w = attn.sum(2) #[b,1,t]
    if self.use_sdp:
      l_length = self.dp(x, x_mask, w, g=g)
      l_length = l_length / torch.sum(x_mask)
    else:
      logw_ = torch.log(w + 1e-6) * x_mask
      logw = self.dp(x, x_mask, g=g)
      l_length = torch.sum((logw - logw_)**2, [1,2]) / torch.sum(x_mask) # for averaging 
    # expand prior
    m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
    logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)
    
    #z_cat = torch.cat((z,ep_expand),dim=1)
    z_slice, ids_slice = commons.rand_slice_segments(z, y_lengths, self.segment_size)
    o = self.dec(z_slice, g=g)
    return o, l_length, attn, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q, e, log_q_e, q_e)

  def infer(self, x, x_lengths, y, y_lengths, sid, y2=None, y2_lengths=None, temp=1.0, noise_scale=0.67, length_scale=1, noise_scale_w=0.8, max_len=None):
    """
    """
    label=None
    gs = self.emb_gs(sid).unsqueeze(-1) # [b, h, 1]

    x, x_mask = self.enc_p(x, x_lengths)
    m_p, logs_p= self.proj_p(x, x_mask)
    eu, e, log_q_e, q_e, y_mask = self.enc_q(y, y_lengths, gs, temp, x2=y2, x2_lengths=y2_lengths, infer=True)
    g = gs+eu
    if self.use_sdp:
      logw = self.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
    else:
      logw = self.dp(x, x_mask, g=g)
    w = torch.exp(logw) * x_mask * length_scale
    w_ceil = torch.ceil(w)
    y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
    y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
    attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
    attn = commons.generate_path(w_ceil, attn_mask)
    m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
    logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
    z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
    z = self.flow(z_p, y_mask, g=g, reverse=True)
    
    o = self.dec((z * y_mask)[:,:,:max_len], g=g)
    return o, attn, y_mask, (z, z_p, m_p, logs_p), eu, e, log_q_e, q_e

  def voice_conversion(self, y, y_lengths, y1, y1_lengths, sid_src, sid_trg):
    assert self.n_speakers > 0, "n_speakers have to be larger than 0."
    g_src = self.emb_gs(sid_src).unsqueeze(-1)
    g_trg = self.emb_gs(sid_trg).unsqueeze(-1)
    
    z, m_q, logs_q, eu_src, e, log_q_e, q_e, y_mask = self.enc_q(y, y_lengths, g=g_src, temp=1.0)
    g_src = g_src + eu_src

    _, _, _, eu_trg, _, _, _, _ = self.enc_q(y1, y1_lengths, g=g_trg, temp=1.0)
    g_trg = g_trg + eu_trg
    
    z_p = self.flow(z, y_mask, g=g_src)

    z_hat = self.flow(z_p, y_mask, g=g_trg, reverse=True)
    o_hat = self.dec(z_hat * y_mask, g=g_trg)
    return o_hat, y_mask, (z, z_p, z_hat)

