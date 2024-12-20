import copy
import logging
import os

import hydra
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader

from models.callbacks import EMAWeightUpdate
from models.diffusion import DDPM, DDPMv2, DDPMWrapper2, SuperResModel, UNetModel
from models.vae import VAE
from util import configure_device, get_dataset
import torch
import torch.nn.functional as F

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

logger = logging.getLogger(__name__)

def __parse_str(s):
    split = s.split(",")
    return [int(s) for s in split if s != "" and s is not None]

@hydra.main(config_path="configs/dataset/cifar10", config_name="train")
def train(config):
    # Get config and setup
    
    config2 = config.vae
    config = config.ddpm

    logger.info(OmegaConf.to_yaml(config))

    # Set seed
    seed_everything(config.training.seed, workers=True)

    
    # Dataset
    root = config.data.root
    d_type = config.data.name
    image_size = config.data.image_size
    dataset = get_dataset(
        d_type, root, image_size, norm=config.data.norm, flip=config.data.hflip
    )
    N = len(dataset)
    batch_size = int(config.training.batch_size)
    batch_size = min(N, batch_size)
    
    

    # VAE 및 DDPM 초기화
    vae = VAE(
        input_res=config2.data.image_size,
        enc_block_str=config2.model.enc_block_config,
        dec_block_str=config2.model.dec_block_config,
        enc_channel_str=config2.model.enc_channel_config,
        dec_channel_str=config2.model.dec_channel_config,
        lr=config2.training.lr,
        alpha=config2.training.alpha,
    )
    
    attn_resolutions = __parse_str(config.model.attn_resolutions)
    dim_mults = __parse_str(config.model.dim_mults)
    decoder_cls = UNetModel if config.training.type == "uncond" else SuperResModel
    
    decoder = decoder_cls(
        in_channels=config.data.n_channels,
        model_channels=config.model.dim,
        out_channels=3,
        num_res_blocks=config.model.n_residual,
        attention_resolutions=attn_resolutions,
        channel_mult=dim_mults,
        use_checkpoint=False,
        dropout=config.model.dropout,
        num_heads=config.model.n_heads,
        z_dim=config.training.z_dim,
        use_scale_shift_norm=config.training.z_cond,
        use_z=config.training.z_cond,
    )
    
    # EMA parameters are non-trainable
    ema_decoder = copy.deepcopy(decoder)
    for p in ema_decoder.parameters():
        p.requires_grad = False
        
        
    ddpm_type = config.training.type
    ddpm_cls = DDPM if ddpm_type == "form2" else DDPM
    
    online_ddpm = ddpm_cls(
        decoder,
        beta_1=config.model.beta1,
        beta_2=config.model.beta2,
        T=config.model.n_timesteps,
    )
    
    target_ddpm = ddpm_cls(
        ema_decoder,
        beta_1=config.model.beta1,
        beta_2=config.model.beta2,
        T=config.model.n_timesteps,
    )
    
    for p in vae.parameters():
        p.requires_grad = False

    assert isinstance(online_ddpm, ddpm_cls)
    assert isinstance(target_ddpm, ddpm_cls)
    logger.info(f"Using DDPM with type: {ddpm_cls} and data norm: {config.data.norm}")
    
    # Joint Training Wrapper
    joint_model = DDPMWrapper2(
        online_ddpm,
        target_ddpm,
        #target_network=None,  # EMA 비활성화
        vae=vae,
        lr=config.training.lr,
        vae_loss_weight=0.1,  # VAE 손실 가중치 초기값
        cfd_rate=config.training.cfd_rate,
        n_anneal_steps=config.training.n_anneal_steps,
        loss=config.training.loss,
        conditional=False if ddpm_type == "uncond" else True,
        grad_clip_val=config.training.grad_clip,
        z_cond=config.training.z_cond,
    )


    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Trainer 초기화
    trainer = pl.Trainer(
        max_epochs=config.training.epochs,
        #accelerator="gpu",  # CPU를 사용
        #devices=1,           # 사용할 CPU 디바이스 개수 (기본적으로 1)
        gpus=1,
        precision=16, #if config.training.fp16 else 32,
        default_root_dir=config.training.results_dir,
    )
    
    results_dir = config.training.results_dir
    chkpt_callback = ModelCheckpoint(
        dirpath=os.path.join(results_dir, "checkpoints"),
        filename=f"ddpmv2-{config.training.chkpt_prefix}" + "-{epoch:02d}-{loss:.4f}",
        every_n_epochs=config.training.chkpt_interval,
        save_on_train_epoch_end=True,
    )
    
    # VAE Checkpoint 설정
    results_dir = config2.training.results_dir
    vae_chkpt_callback = ModelCheckpoint(
        dirpath=os.path.join(results_dir, "checkpoints"),
        filename="vae-{epoch:02d}-{train_loss:.4f}",
        every_n_epochs=config2.training.chkpt_interval,  # VAE 저장 주기
        save_top_k=-1,  # 모든 에포크를 저장
        #save_weights_only=True,  # VAE 가중치만 저장
    )
    
    loader = DataLoader(
        dataset,
        batch_size,
        num_workers=config.training.workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        #**loader_kws,
        
    )
    
    # 학습 종료 후 Latent Space 저장
    vae.save_latent_space(loader, save_path=os.path.join(config.training.results_dir, "latent_space.pt"))
    
    
    train_kwargs = {}
    train_kwargs["callbacks"] = [chkpt_callback, vae_chkpt_callback]
    ema_callback = EMAWeightUpdate(tau=config.training.ema_decay)
    train_kwargs["callbacks"].append(ema_callback)
    
    
        
    trainer = pl.Trainer(
        callbacks = train_kwargs["callbacks"], #, vae_chkpt_callback
        default_root_dir = config.training.results_dir,
        max_epochs = config.training.epochs,
        log_every_n_steps = config.training.log_step
    )
    
    trainer.fit(joint_model, train_dataloaders=loader)


if __name__ == "__main__":
    train()
