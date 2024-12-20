#python /home/dsail/leejonghyun/deepgeneratmodel/DiffuseVAE-main/main/test.py reconstruct --device cuda:0 \
#                                --dataset cifar10 \
#                                --image-size 32 \
#                                --save-path ~/vae_cifar10_recons_step2_latentspace/ \
#                                --write-mode image \
#                                "/home/dsail/leejonghyun/deepgeneratmodel/DiffuseVAE-main/final/result/checkpoints/vae--epoch=999-train_loss=0.0000.ckpt" \
#                                "/home/dsail/leejonghyun/deepgeneratmodel/DiffuseVAE-main/cifar10/"

# python main/test.py reconstruct --device gpu:0 \
#                                 --dataset ffhq \
#                                 --image-size 128 \
#                                 --num-samples 64 \
#                                 --save-path ~/vae_samples_ffhq128_deletem_recons/ \
#                                 --write-mode image \
#                                 /data1/kushagrap20/vae_ffhq128_11thJune_alpha\=1.0/checkpoints/vae-ffhq128_11thJune_alpha\=1.0-epoch\=496-train_loss\=0.0000.ckpt \
#                                 ~/datasets/ffhq/

python /home/dsail/leejonghyun/deepgeneratmodel/DiffuseVAE-main/main/test.py sample --device cuda:0 \
                           --image-size 32 \
                           --seed 0 \
                           --num-samples 50000 \
                           --save-path ~/vae_cifar10_recons_step2_latentspace/ \
                           --write-mode image \
                           512 \
                           /home/dsail/leejonghyun/deepgeneratmodel/DiffuseVAE-main/final/result/checkpoints/vae-epoch=140-train_loss=0.0000.ckpt \

# python main/test.py sample --device gpu:0 \
#                             --image-size 128 \
#                             --seed 0 \
#                             --num-samples 64 \
#                             --save-path ~/vae_samples_ffhq128_deletem/ \
#                             --write-mode image \
#                             1024 \
#                             /data1/kushagrap20/vae_ffhq128_11thJune_alpha\=1.0/checkpoints/vae-ffhq128_11thJune_alpha\=1.0-epoch\=496-train_loss\=0.0000.ckpt \


# python main/test.py reconstruct --device gpu:0 \
#                                 --dataset afhq \
#                                 --image-size 128 \
#                                 --save-path ~/reconstructions/afhq_reconsv2/ \
#                                 --write-mode numpy \
#                                 ~/vae_afhq_alpha\=1.0/checkpoints/vae-afhq_alpha=1.0-epoch=1499-train_loss=0.0000.ckpt \
#                                 ~/datasets/afhq/

# python main/test.py sample --device gpu:0 \
#                             --image-size 128 \
#                             --seed 0 \
#                             --num-samples 64 \
#                             --save-path ~/afhq_vae_samples1/ \
#                             --write-mode image \
#                             1024 \
#                             ~/vae_afhq_alpha\=1.0/checkpoints/vae-afhq_alpha=1.0-epoch=1499-train_loss=0.0000.ckpt \

# python main/test.py reconstruct --device gpu:0 \
#                            --dataset celebamaskhq \
#                            --num-samples 16 \
#                            --save-path ~/vae_alpha_1_0_samples/ \
#                            ~/checkpoints/cmhq/vae-epoch\=189-train_loss\=0.00.ckpt \
#                            ~/datasets/CelebAMask-HQ/