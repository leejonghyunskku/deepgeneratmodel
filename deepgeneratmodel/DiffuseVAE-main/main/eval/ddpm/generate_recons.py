# Helper script to generate reconstructions from a conditional DDPM model
# Add project directory to sys.path
import os
import sys

p = os.path.join(os.path.abspath("."), "/home/dsail/leejonghyun/deepgeneratmodel/DiffuseVAE-main/main")
sys.path.insert(1, p)

import copy

import hydra
import pytorch_lightning as pl
import torch
from models.callbacks import ImageWriter
from models.diffusion import DDPM, DDPMv2, DDPMv3, DDPMWrapper2, DDPMWrapper3, SuperResModel, UNetModel
from models.vae import VAE
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader
from util import configure_device, get_dataset

from datasets import CIFAR10Dataset

p = os.path.join(os.path.abspath("."), "/home/dsail/leejonghyun/deepgeneratmodel/DiffuseVAE-main/main")
sys.path.insert(1, p)

def __parse_str(s):
    split = s.split(",")
    return [int(s) for s in split if s != "" and s is not None]

#@hydra.main(config_path='home/dsail/leejonghyun/deepgeneratmodel/DiffuseVAE-main/main/configs/dataset/cifar10' ,config_name="test")
@hydra.main(config_path=os.path.join(p, "configs/dataset/cifar10"), config_name="test", version_base="1.1")
def generate_recons(config):
    config_ddpm = config.dataset.ddpm
    config_vae = config.dataset.vae
    seed_everything(config_ddpm.evaluation.seed, workers=True)

    print(config_ddpm)
    batch_size = config_ddpm.evaluation.batch_size
    n_steps = config_ddpm.evaluation.n_steps
    n_samples = config_ddpm.evaluation.n_samples
    image_size = config_ddpm.data.image_size
    ddpm_latent_path = config_ddpm.data.ddpm_latent_path
    ddpm_latents = torch.load(ddpm_latent_path) if ddpm_latent_path != "" else None

    # Load pretrained VAE
    print("####################################################################################")
    vae = VAE.load_from_checkpoint(
        config_vae.evaluation.chkpt_path,
        input_res=image_size,
        enc_block_str="32x7,32d2,32t16,16x4,16d2,16t8,8x4,8d2,8t4,4x3,4d4,4t1,1x3",
        dec_block_str="1x1,1u4,1t4,4x2,4u2,4t8,8x3,8u2,8t16,16x7,16u2,16t32,32x15",
        enc_channel_str="32:64,16:128,8:256,4:256,1:512",
        dec_channel_str="32:64,16:128,8:256,4:256,1:512", strict=False,
        #enc_block_str = "128x1,128d2,128t64,64x3,64d2,64t32,32x3,32d2,32t16,16x7,16d2,16t8,8x3,8d2,8t4,4x3,4d4,4t1,1x2",
        #enc_channel_str = "128:64,64:64,32:128,16:128,8:256,4:512,1:1024",

        #dec_block_str = "1x1,1u4,1t4,4x2,4u2,4t8,8x2,8u2,8t16,16x6,16u2,16t32,32x2,32u2,32t64,64x2,64u2,64t128,128x1",
        #dec_channel_str = "128:64,64:64,32:128,16:128,8:256,4:512,1:1024", strict=False,
    )
    vae.eval()
    print("####################################################################################")
    # Load pretrained wrapper
    attn_resolutions = __parse_str(config_ddpm.model.attn_resolutions)
    dim_mults = __parse_str(config_ddpm.model.dim_mults)
    
    decoder = SuperResModel(
        in_channels=config_ddpm.data.n_channels,
        model_channels=config_ddpm.model.dim,
        out_channels=3,
        num_res_blocks=config_ddpm.model.n_residual,
        attention_resolutions=attn_resolutions,
        channel_mult=dim_mults,
        use_checkpoint=False,
        dropout=config_ddpm.model.dropout,
        num_heads=config_ddpm.model.n_heads,
        z_dim=config_ddpm.evaluation.z_dim,
        use_scale_shift_norm=config_ddpm.evaluation.z_cond,
        use_z=config_ddpm.evaluation.z_cond,
    )

    ema_decoder = copy.deepcopy(decoder)
    decoder.eval()
    ema_decoder.eval()

    ddpm_cls = DDPMv3 if config_ddpm.evaluation.type == "form2" else DDPMv3
    online_ddpm = ddpm_cls(
        decoder,
        beta_1=config_ddpm.model.beta1,
        beta_2=config_ddpm.model.beta2,
        T=config_ddpm.model.n_timesteps,
        var_type=config_ddpm.evaluation.variance,
    )
    target_ddpm = ddpm_cls(
        ema_decoder,
        beta_1=config_ddpm.model.beta1,
        beta_2=config_ddpm.model.beta2,
        T=config_ddpm.model.n_timesteps,
        var_type=config_ddpm.evaluation.variance,
    )
    print("####################################################################################")
    ddpm_wrapper = DDPMWrapper3.load_from_checkpoint(
        
        config_ddpm.evaluation.chkpt_path,
        #target_network=None,
        online_network=online_ddpm,
        target_network=target_ddpm,
        vae=vae,
        conditional=True,
        pred_steps=n_steps,
        eval_mode="recons",
        resample_strategy=config_ddpm.evaluation.resample_strategy,
        skip_strategy=config_ddpm.evaluation.skip_strategy,
        sample_method=config_ddpm.evaluation.sample_method,
        sample_from=config_ddpm.evaluation.sample_from,
        data_norm=config_ddpm.data.norm,
        temp=config_ddpm.evaluation.temp,
        guidance_weight=config_ddpm.evaluation.guidance_weight,
        z_cond=config_ddpm.evaluation.z_cond,
        ddpm_latents=ddpm_latents,
        strict=False,
    )
    print("####################################################################################")
    print("####################################################################################")
    print("####################################################################################")
    print("####################################################################################")
    print("####################################################################################")
    print("####################################################################################")
    print("####################################################################################")
    print("####################################################################################")
    print("####################################################################################")
    print("####################################################################################")
    print("####################################################################################")
    print("####################################################################################")
    # Dataset
    root = config_ddpm.data.root
    d_type = config_ddpm.data.name
    image_size = config_ddpm.data.image_size
    dataset = get_dataset(
        d_type,
        root,
        image_size,
        norm=config_ddpm.data.norm,
        flip=config_ddpm.data.hflip,
        subsample_size=n_samples,
    )

    # Setup devices
    test_kwargs = {}
    loader_kws = {}
    device = config_ddpm.evaluation.device
    if device.startswith("gpu"):
        _, devs = configure_device(device)
        test_kwargs["gpus"] = devs

        # Disable find_unused_parameters when using DDP training for performance reasons
        loader_kws["persistent_workers"] = True
    elif device == "tpu":
        test_kwargs["tpu_cores"] = 8

    # Predict loader
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=False,
        pin_memory=True,
        shuffle=False,
        num_workers=config_ddpm.evaluation.workers,
        **loader_kws,
    )

    # Predict trainer
    write_callback = ImageWriter(
        config_ddpm.evaluation.save_path,
        "batch",
        n_steps=n_steps,
        eval_mode="recons",
        conditional=True,
        sample_prefix=config_ddpm.evaluation.sample_prefix,
        save_mode=config_ddpm.evaluation.save_mode,
        save_vae=config_ddpm.evaluation.save_vae,
        is_norm=config_ddpm.data.norm,
    )

    test_kwargs["callbacks"] = [write_callback]
    test_kwargs["default_root_dir"] = config_ddpm.evaluation.save_path
    trainer = pl.Trainer(**test_kwargs)
    trainer.predict(ddpm_wrapper, val_loader)


if __name__ == "__main__":
    generate_recons()
