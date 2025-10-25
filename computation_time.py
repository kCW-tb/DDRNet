import torch
import time
from DDRNet import DDRNet
from thop import profile # FLOPs 계산을 위한 라이브러리
import statistics as stats # 중앙값 계산을 위함

# --- 벤치마킹 설정 ---
modelName = "DDRNet"
numClasses = 19
compute_speed = True # 속도 측정을 실행할지 여부

# cuDNN 벤치마크 모드 활성화 (입력 크기가 고정일 때 성능 향상)
torch.backends.cudnn.benchmark = True
device = torch.device('cuda')
# 모델 이름을 기반으로 모델 객체를 생성하고, 평가 모드로 설정 후 GPU로 이동
model = eval(modelName)(num_classes=numClasses).eval().to(device)

# --- 1. 모델 파라미터 수 계산 ---
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params / 1024 **2:.3f} M")

# 모델 입력을 위한 더미 텐서 생성 (Batch=1, Channel=3, Height=1080, Width=1920)
x = torch.randn(1, 3, 1080, 1920, device=device)

# --- 2. FLOPs 및 파라미터 수 계산 (thop 라이브러리 사용) ---
# profile 함수를 통해 모델의 연산량(FLOPs)과 파라미터 수를 계산
flops, params = profile(model, inputs=(x,))
print(f'Parameters: {params/(1024*1024):.2f} M')
result = f'FLOPs: {flops/(1024*1024*1024):.2f} G'
print(result)

# 측정된 FLOPs 결과를 파일에 저장
with open("FLOPs.txt", "w") as f:
    f.write(result + "\n")

# --- 3. 최대 GPU 메모리 사용량 측정 ---
torch.cuda.reset_peak_memory_stats() # 메모리 통계 초기화
with torch.no_grad(): # 그래디언트 계산 비활성화
    _ = model(x) # 모델 순전파 실행
torch.cuda.synchronize() # GPU 작업 완료 대기
# 최대 할당된 메모리를 MB 단위로 계산
mem_used = torch.cuda.max_memory_allocated() / (1024**2)
print(f"Peak GPU Memory: {mem_used:.2f} MB")

# --- 4. 추론 속도 (Latency, FPS) 측정 ---
if compute_speed:
    # 워밍업: 초기 오버헤드를 없애기 위해 여러 번 실행
    with torch.no_grad(): # 개선점: torch.inference_mode()가 더 효율적
        for _ in range(50):
            _ = model(x)
    torch.cuda.synchronize()

    results = []
    iterations = 200 # 각 측정 주기마다 반복할 횟수
    
    # 총 9번의 측정 주기를 반복하여 신뢰성 있는 데이터 수집
    for _ in range(9):  
        # 정확한 시간 측정을 위해 시작점과 끝점에 synchronize 호출
        torch.cuda.synchronize()
        t_start = time.time()
        
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(x)
                
        torch.cuda.synchronize()
        t_end = time.time()

        elapsed = t_end - t_start # 총 소요 시간
        # 단일 추론에 대한 평균 지연 시간(latency)을 ms 단위로 계산
        latency_ms = (elapsed / iterations) * 1000
        results.append(latency_ms)

    # --- 결과 통계 계산 및 출력 ---
    mean_ms = sum(results) / len(results) # 9번 측정한 latency의 평균값
    median_ms = stats.median(results)    # 9번 측정한 latency의 중앙값 (더 안정적인 지표)
    
    # 평균 및 중앙값 latency를 기반으로 FPS(초당 프레임 수) 계산
    fps_mean = 1000.0 / mean_ms
    fps_median = 1000.0 / median_ms

    print("\n========= Speed (per forward) =========")
    print(f"Per-run latencies (ms): {[round(v,3) for v in results]}")
    # 이상치에 강한 중앙값을 기준으로 최종 성능을 리포트
    print(f"Median: {median_ms:.3f} ms   (~{fps_median:.2f} FPS)")