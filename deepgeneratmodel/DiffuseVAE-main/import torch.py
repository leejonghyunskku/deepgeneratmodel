import torch

checkpoint = torch.load("C:/Users/sigol/Desktop/skku/2학기수업/심층생성모델/프로젝트/DiffuseVAE-main/result/checkpoints/vae--epoch=09-train_loss=0.0000.ckpt", map_location="cpu")
print(checkpoint.keys())  # 저장된 키 확인
print(checkpoint["hyper_parameters"])  # 하이퍼파라미터 정보 출력
