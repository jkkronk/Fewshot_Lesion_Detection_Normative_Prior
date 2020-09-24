import torch
import torch.nn as nn
import torch.optim as optim

def MAP_unet(input_img, dec_mu, vae_model, unet_model, riter, device, weight = 1, step_size=0.003, writer = None):
    # Init params
    input_img = input_img
    dec_mu = dec_mu.to(device).float()
    img_ano = nn.Parameter(input_img.clone().to(device), requires_grad=True)

    unet_model.eval()
    # Iterate until convergence
    for i in range(riter):
        __, z_mean, z_cov, __ = vae_model(img_ano.unsqueeze(1).double())

        kl_loss = -0.5 * torch.sum(1 + z_cov - z_mean.pow(2) - z_cov.exp())
        l2_loss = torch.sum((dec_mu.view(-1, dec_mu.numel()) - img_ano.view(-1, img_ano.numel())).pow(2))

        elbo = l2_loss + kl_loss

        # Gradient of prior with respect to X
        elbo_grad, = torch.autograd.grad(elbo, img_ano,
                                         grad_outputs=elbo.data.new(elbo.shape).fill_(1),
                                         create_graph=True)

        # Segmentation network
        nn_input = torch.stack([input_img, img_ano.detach()]).permute((1, 0, 2, 3)).float().to(device)
        out = unet_model(nn_input).squeeze(1)
        img_grad = elbo_grad.detach() - weight * elbo_grad.detach() * out

        # Gradient step
        img_ano_update = img_ano - step_size * img_grad.to(device)

        img_ano = img_ano_update.detach().to(device)
        img_ano.requires_grad = True

    if not writer == None :
        writer.add_image('Img', normalize_tensor(input_img.unsqueeze(1)[:16]), dataformats='NCHW')
        #writer.add_image('Seg', normalize_tensor(input_seg.unsqueeze(1)[:16]), dataformats='NCHW')
        writer.add_image('Restored', normalize_tensor(img_ano.unsqueeze(1)[:16]), dataformats='NCHW')
        writer.add_image('Restored_Img', normalize_tensor((img_ano - input_img).pow(2).unsqueeze(1)[:16]),
                         dataformats='NCHW')
        writer.add_image('Out', normalize_tensor(out.unsqueeze(1)[:16]), dataformats='NCHW')
        writer.add_image('ELBO', normalize_tensor(elbo_grad.unsqueeze(1)[:16]), dataformats='NCHW')
        writer.add_image('Grad', normalize_tensor(img_grad.unsqueeze(1)[:16]), dataformats='NCHW')

        writer.flush()
    return img_ano

def normalize_tensor(input_tens):
    i_max = input_tens.max()
    i_min = input_tens.min()
    input_tens = (input_tens-i_min)/(i_max-i_min)
    return input_tens
