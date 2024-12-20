# fidelity --gpu 3 --fid --input1 ~/datasets/CelebAMask-HQ-128/ --input2 ~/ddpm_cmhq_confirm_form2/49/images/ --input1-cache-name cmhq128
fidelity --gpu 0 --fid --input1 /home/dsail/leejonghyun/deepgeneratmodel/DiffuseVAE-main/final2/1000/images/ --input2 cifar10-train

fidelity --gpu 0 --isc --input1 /home/dsail/leejonghyun/deepgeneratmodel/DiffuseVAE-main/final2/1000/images/ --input2 cifar10-train

# fidelity --gpu 3 --fid --input1 ~/datasets/img_align_celeba_64/ --input2 ~/ddpm_celeba64_benchmark_speedvsquality/form2_ddim/310/1000/images/ --input1-cache-name celeba64
# fidelity --gpu 3 --fid --input1 ~/datasets/celeba_hq_256/ --input2 ~/ddpm_chq256_benchmark_speedvsquality/form1_ddim_expde/96/images/ --input1-cache-name chq256
fidelity --gpu 0 --fid --input1 /home/dsail/leejonghyun/deepgeneratmodel/DiffuseVAE-main/train_ae_result_vae/ --input2 /home/dsail/leejonghyun/deepgeneratmodel/DiffuseVAE-main/train_ae_result_orig/