import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def get_brightness_values(directory, sampling_rate=1):
    """
    지정된 디렉토리의 이미지에서 밝기 값을 추출합니다.
    sampling_rate: N 값. N개의 파일당 1개의 파일만 처리합니다. (기본값=1, 모두 처리)
    """
    brightness_values = []
    image_count = 0
    file_index = 0
    print(f"'{directory}' 경로에서 이미지 파일을 스캔합니다 (샘플링 비율: {sampling_rate}당 1개)...")

    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.png')):
                file_index += 1
                # --- 💡 주요 변경 사항: 샘플링 로직 ---
                if file_index % sampling_rate != 0:
                    continue # sampling_rate에 해당하지 않으면 건너뛰기
                # ------------------------------------

                image_path = os.path.normpath(os.path.join(root, file))
                try:
                    with Image.open(image_path) as img:
                        grayscale_img = img.convert('L')
                        brightness_values.extend(list(grayscale_img.getdata()))
                        image_count += 1
                except Exception as e:
                    print(f"파일 처리 오류 {image_path}: {e}")

    print(f"'{directory}'에서 총 {file_index}개 파일 발견, {image_count}개 샘플링하여 {len(brightness_values):,}개 픽셀 수집 완료.")
    return brightness_values


if __name__ == "__main__":
    image_root = 'data\\image'
    train_dir = os.path.join(image_root, 'train')
    test_dir = os.path.join(image_root, 'test')

    # Train 데이터는 5장당 1장씩 샘플링, Test는 전부 처리
    train_brightness = get_brightness_values(train_dir, sampling_rate=10)
    test_brightness = get_brightness_values(test_dir, sampling_rate=1) # 기본값
    
    if not train_brightness and not test_brightness:
        print("분석할 이미지를 찾을 수 없습니다. 경로를 확인해주세요.")
    else:
        # --- 💡 주요 변경 사항: 그래프 분리 ---
        # 2개의 세로 플롯을 생성
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))
        
        # 전체 제목 설정
        fig.suptitle('Train vs Test Image Brightness Distribution', fontsize=16)

        # 첫 번째 그래프 (Train)
        axes[0].hist(train_brightness, bins=256, range=(0, 256), color='blue', alpha=0.7)
        axes[0].set_title(f'Train Data ({len(train_brightness):,} pixels)')
        axes[0].set_xlabel('Brightness Value')
        axes[0].set_ylabel('Pixel Frequency')
        axes[0].grid(axis='y', linestyle='--', alpha=0.7)
        
        # 두 번째 그래프 (Test)
        axes[1].hist(test_brightness, bins=256, range=(0, 256), color='orange', alpha=0.7)
        axes[1].set_title(f'Test Data ({len(test_brightness):,} pixels)')
        axes[1].set_xlabel('Brightness Value (0=Black, 255=White)')
        axes[1].set_ylabel('Pixel Frequency')
        axes[1].grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # 전체 제목과 겹치지 않도록 레이아웃 조정
        
        plt.savefig('brightness_separated_plots.png')
        print("\n분석 그래프를 'brightness_separated_plots.png' 파일로 저장했습니다.")
        plt.show()