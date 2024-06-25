import subprocess

if __name__ == "__main__":
    # 파이프라인을 테스트하고 컴파일
    #subprocess.run(["python", "test_pipeline.py"], check=True)

    # 실제 실행 단계는 주석 처리하여 테스트 시 실행되지 않도록 합니다.
     subprocess.run(["python", "data_generation.py"], check=True)
     subprocess.run(["python", "model_training.py"], check=True)
     subprocess.run(["python", "tapestry_dht.py"], check=True)
     subprocess.run(["python", "smartthings_integration.py"], check=True)
