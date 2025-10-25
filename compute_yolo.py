# import torch
# import time
# from thop import profile
# import statistics as stats
# from ultralytics import YOLO
# import argparse

# def benchmark(args):
#     compute_speed = True

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     torch.backends.cudnn.benchmark = True
#     print(f"Using device: {device}")

#     try:
#         model = YOLO(args.model_path).to(device)
#         model.eval()
#         print(f"Model loaded successfully from {args.model_path}")
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         return

#     total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"Total trainable parameters: {total_params / 1e6:.3f} M")

#     try:
#         x = torch.randn(1, 3, args.img_size, args.img_size, device=device)
#         print(f"Dummy input tensor created with size: 1x3x{args.img_size}x{args.img_size}")
#     except Exception as e:
#         print(f"Error creating input tensor: {e}")
#         return

#     try:
#         flops, params = profile(model.model, inputs=(x,), verbose=False)
#         print(f'Parameters (from thop): {params/1e6:.2f} M')
#         result = f'FLOPs: {flops/1e9:.2f} G'
#         print(result)
#         with open("FLOPs_yolo.txt", "w") as f:
#             f.write(result + "\n")
#     except Exception as e:
#         print(f"Could not calculate FLOPs with thop. Error: {e}")
#         print("Note: Some modern model architectures may not be fully compatible with thop.")

#     if device.type == 'cuda':
#         torch.cuda.reset_peak_memory_stats()
#         with torch.no_grad():
#             _ = model(x)
#         torch.cuda.synchronize()
#         mem_used = torch.cuda.max_memory_allocated() / (1024**2)
#         print(f"Peak GPU Memory: {mem_used:.2f} MB")

#     if compute_speed:
#         print("\nStarting speed benchmark...")
#         with torch.no_grad():
#             for _ in range(50):
#                 _ = model(x)
#         if device.type == 'cuda':
#             torch.cuda.synchronize()

#         results = []
#         iterations = 200
        
#         for i in range(9):
#             if device.type == 'cuda':
#                 torch.cuda.synchronize()
#             t_start = time.time()
            
#             with torch.no_grad():
#                 for _ in range(iterations):
#                     _ = model(x)
                    
#             if device.type == 'cuda':
#                 torch.cuda.synchronize()
#             t_end = time.time()

#             elapsed = t_end - t_start # 총 소요 시간
#             # 단일 추론에 대한 평균 지연 시간(latency)
#             latency_ms = (elapsed / iterations) * 1000
#             results.append(latency_ms)
#             print(f"Run {i+1}/9: {latency_ms:.3f} ms/inference")

#         median_ms = stats.median(results)
#         fps_median = 1000.0 / median_ms
        
#         print("\n========= Speed (per forward) =========")
#         print(f"All latencies (ms): {[round(v,3) for v in results]}")
#         print(f"Median Latency: {median_ms:.3f} ms   (≈{fps_median:.2f} FPS)")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Benchmark a YOLO model.")
#     parser.add_argument("--model_path", type=str, default="yolo11m-seg.yaml",
#                         help="Path to the YOLO model file (.yaml or .pt).")
#     parser.add_argument("--img_size", type=int, default=640,
#                         help="Input image size (height and width).")
#     args = parser.parse_args()
    
#     benchmark(args)