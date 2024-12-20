import os
import subprocess

# 환경 변수 설정
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# fidelity 명령어 실행
command = [
    "fidelity", "--cpu", "--fid",
    "--input1", "C:/Users/sigol/Desktop/skku/2학기수업/심층생성모델/프로젝트/DiffuseVAE-main/결과/original",
    "--input2", "C:/Users/sigol/Desktop/skku/2학기수업/심층생성모델/프로젝트/DiffuseVAE-main/결과/vae"
]

# subprocess를 사용하여 명령어 실행
subprocess.run(command)
