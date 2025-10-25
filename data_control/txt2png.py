import os
import cv2
import numpy as np

# --- 사용자 설정 부분 ---

# 1. 원본 이미지가 있는 폴더 (마스크의 크기를 알아내기 위해 필요)
#IMAGE_DIR = r"C:\DDRNet\data\ETRI\train\images"
#IMAGE_DIR = r"C:\DDRNet\data\ETRI\val\images"
IMAGE_DIR = r"C:\DDRNet\data\ETRI\test\images"

# 2. 변환할 .txt 라벨이 있는 폴더
#TXT_LABEL_DIR = r"C:\DDRNet\data\ETRI\train\labels"
#TXT_LABEL_DIR = r"C:\DDRNet\data\ETRI\val\labels"
TXT_LABEL_DIR = r"C:\DDRNet\data\ETRI\test\labels"

# 3. 변환된 .png 마스크를 저장할 폴더
#OUTPUT_PNG_DIR = r"C:\DDRNet\data\ETRI\train\masks_png"
#OUTPUT_PNG_DIR = r"C:\DDRNet\data\ETRI\val\masks_png"
OUTPUT_PNG_DIR = r"C:\DDRNet\data\ETRI\test\masks_png"

# 4. 클래스 이름/ID와 마스크에 채울 숫자 값을 매핑
#    - DDRNet의 yaml 파일에 설정된 NUM_CLASSES와 맞춰야 합니다.
#    - 배경은 보통 0으로 처리됩니다.
#    - .txt 파일에 클래스 이름(예: car)이 있으면 {'car': 1, 'person': 2} 처럼,
#      클래스 ID(예: 0)가 있으면 {0: 1, 1: 2} 처럼 작성합니다.
#      여기서는 .txt 파일의 ID가 0, 1, 2... 라고 가정하고, 마스크 값도 동일하게 1:1 매핑합니다.
# 4. 클래스 이름/ID와 마스크에 채울 숫자 값을 매핑
CLASS_TO_ID = {str(i): i for i in range(42)}

# --- 코드 본문 (수정 필요 없음) ---

def convert_txt_to_png():
    """
    .txt 폴리곤 라벨을 .png 마스크 이미지로 변환합니다.
    """
    os.makedirs(OUTPUT_PNG_DIR, exist_ok=True)
    
    txt_files = [f for f in os.listdir(TXT_LABEL_DIR) if f.endswith('.txt')]
    print(f"총 {len(txt_files)}개의 .txt 파일을 변환합니다.")

    processed_count = 0
    for txt_filename in txt_files:
        base_filename = os.path.splitext(txt_filename)[0]
        
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png']:
            potential_path = os.path.join(IMAGE_DIR, base_filename + ext)
            if os.path.exists(potential_path):
                img_path = potential_path
                break
        
        if not img_path:
            print(f"경고: {base_filename}에 해당하는 원본 이미지를 찾을 수 없습니다. 건너뜁니다.")
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            print(f"경고: 이미지 파일 {img_path}를 읽을 수 없습니다. 건너뜁니다.")
            continue
        height, width, _ = img.shape
        
        mask = np.zeros((height, width), dtype=np.uint8)
        
        txt_path = os.path.join(TXT_LABEL_DIR, txt_filename)
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                
                class_id_str = parts[0]
                
                if class_id_str not in CLASS_TO_ID:
                    print(f"경고: {txt_filename}의 '{class_id_str}' 클래스를 CLASS_TO_ID에서 찾을 수 없습니다.")
                    continue
                mask_value = CLASS_TO_ID[class_id_str]
                
                if len(parts[1:]) % 2 != 0:
                    print(f"경고: {txt_filename}에 홀수 개의 좌표값이 있습니다. 건너뜁니다.")
                    continue

                # =================== 이 부분이 수정되었습니다 ===================
                try:
                    # 1. 좌표를 소수(float)로 읽어들입니다.
                    normalized_coords = np.array(parts[1:], dtype=np.float32).reshape((-1, 2))

                    # 2. 이미지의 너비(width)와 높이(height)를 곱하여 실제 픽셀 좌표로 변환합니다.
                    pixel_coords = (normalized_coords * np.array([width, height])).astype(np.int32)
                
                except ValueError:
                    print(f"경고: {txt_filename} 파일의 좌표값에 숫자가 아닌 값이 포함되어 있습니다. 해당 줄을 건너뜁니다.")
                    continue
                # ==========================================================
                
                cv2.fillPoly(mask, [pixel_coords], color=mask_value)
    
        output_path = os.path.join(OUTPUT_PNG_DIR, base_filename + '.png')
        cv2.imwrite(output_path, mask)
        processed_count += 1
        if processed_count % 100 == 0:
            print(f"{processed_count}/{len(txt_files)} 파일 처리 완료...")

    print(f"\n변환 완료! 총 {processed_count}개의 파일을 {OUTPUT_PNG_DIR} 폴더에 저장했습니다.")


if __name__ == '__main__':
    convert_txt_to_png()