import torch 
from torch.nn import functional as F
import numpy as np
import commons


def feature_loss(fmap_r, fmap_g):
  loss = 0
  for dr, dg in zip(fmap_r, fmap_g):
    for rl, gl in zip(dr, dg):
      rl = rl.float().detach()
      gl = gl.float()
      loss += torch.mean(torch.abs(rl - gl))

  return loss * 2 


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
  loss = 0
  r_losses = []
  g_losses = []
  for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
    dr = dr.float()
    dg = dg.float()
    r_loss = torch.mean((1-dr)**2)
    g_loss = torch.mean(dg**2)
    loss += (r_loss + g_loss)
    r_losses.append(r_loss.item())
    g_losses.append(g_loss.item())

  return loss, r_losses, g_losses


def generator_loss(disc_outputs):
  loss = 0
  gen_losses = []
  for dg in disc_outputs:
    dg = dg.float()
    l = torch.mean((1-dg)**2)
    gen_losses.append(l)
    loss += l

  return loss, gen_losses


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
  """
  z_p, logs_q: [b, h, t_t]
  m_p, logs_p: [b, h, t_t]
  """
  z_p = z_p.float()
  logs_q = logs_q.float()
  m_p = m_p.float()
  logs_p = logs_p.float()
  z_mask = z_mask.float()

  kl = logs_p - logs_q - 0.5
  kl += 0.5 * ((z_p - m_p)**2) * torch.exp(-2. * logs_p)
  kl = torch.sum(kl * z_mask)
  l = kl / torch.sum(z_mask)
  return l

def log_normal(x, mu, var):
  """Logarithm of normal distribution with mean=mu and variance=var
     log(x|μ, σ^2) = loss = -0.5 * Σ log(2π) + log(σ^2) + ((x - μ)/σ)^2
  Args:
     x: (array) corresponding array containing the input
     mu: (array) corresponding array containing the mean 
     var: (array) corresponding array containing the variance
  Returns:
     output: (array/float) depending on average parameters the result will be the mean
                            of all the sample losses or an array with the losses per sample
  """
  eps = 1e-8
  var = var + eps
  return -0.5 * torch.sum(np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var, dim=-1)


def gaussian_loss(mu, var2):
  """Variational loss when using labeled data without considering reconstruction loss 
    mu2, sigma2 = y, B
    A = log(sigma2/sigma1) + 1/(2sigma2**2)(sigma1**2 + (mu1-mu2)**2)-1/2 
  Returns:
     output: (array/float) depending on average parameters the result will be the mean
                            of all the sample losses or an array with the losses per sample
  """
  eps = 1e-8
  var2 = var2 + eps
  return -0.5 * torch.sum(1 + torch.log(var2) - mu.pow(2) - var2)

def entropy(log_q_y, q_y, num_class):
  """Entropy loss
      loss = (1/n) * -Σ targets*log(predicted)
  Args:
      logits: (array) corresponding array containing the logits of the categorical variable
      real: (array) corresponding array containing the true labels
 
  Returns:
      output: (array/float) depending on average parameters the result will be the mean
                            of all the sample losses or an array with the losses per sample
  """
  #log_q_y_ = log_q_y.view(64,16,10) 
  #print("argmax num_class:",torch.argmax(log_q_y_,dim=2))
  #print("argmax class_dim:",torch.argmax(log_q_y_,dim=1))
  kl = q_y * (log_q_y - np.log(1.0/num_class))
  #kl = kl.reshape([-1, 16, num_class])
  kl = kl.sum(-1)
  return kl.mean()

def classify_loss(predict, target):
    #loss_func = torch.nn.CrossEntropyLoss()
    loss_func = torch.nn.NLLLoss()
    output = loss_func(torch.log(predict), target)
    return output
