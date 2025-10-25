# #마스크 원본 이미지에 시각화 코드

# import os
# import cv2
# import numpy as np
# from PIL import Image

# # --- 사용자 설정 부분 ---

# # 1. 변환된 .png 마스크가 있는 폴더
# PNG_MASK_DIR = r"C:\DDRNet\data\ETRI\train\labels" # 이전에 생성한 .png 마스크 폴더

# # 2. 시각화된 컬러 마스크를 저장할 폴더
# OUTPUT_VIS_DIR = r"C:\DDRNet\data\ETRI\train\masks_visualized"

# # 3. 클래스 ID에 매핑할 색상 팔레트 정의 (RGB 값)
# #    - NUM_CLASSES: 42에 맞춰 42가지 색상을 정의해야 합니다.
# #    - 0번 인덱스는 보통 배경 (검은색 또는 어두운 색)
# #    - 다른 색상들은 서로 잘 구별될 수 있도록 다채롭게 구성하는 것이 좋습니다.
# #    - 여기서는 임의의 색상들을 사용하며, 필요에 따라 더 좋은 팔레트로 교체하세요.
# COLOR_PALETTE = [
#     (0, 0, 0),       # 0: 배경 (Black)
#     (128, 0, 0),     # 1: Dark Red
#     (0, 128, 0),     # 2: Dark Green
#     (128, 128, 0),   # 3: Dark Yellow
#     (0, 0, 128),     # 4: Dark Blue
#     (128, 0, 128),   # 5: Dark Magenta
#     (0, 128, 128),   # 6: Dark Cyan
#     (128, 128, 128), # 7: Gray
#     (64, 0, 0),      # 8:
#     (192, 0, 0),     # 9:
#     (64, 128, 0),    # 10:
#     (192, 128, 0),   # 11:
#     (64, 0, 128),    # 12:
#     (192, 0, 128),   # 13:
#     (64, 128, 128),  # 14:
#     (192, 128, 128), # 15:
#     (0, 64, 0),      # 16:
#     (128, 64, 0),    # 17:
#     (0, 192, 0),     # 18:
#     (128, 192, 0),   # 19:
#     (0, 64, 128),    # 20:
#     (128, 64, 128),  # 21:
#     (0, 192, 128),   # 22:
#     (128, 192, 128), # 23:
#     (64, 64, 0),     # 24:
#     (192, 64, 0),    # 25:
#     (64, 192, 0),    # 26:
#     (192, 192, 0),   # 27:
#     (64, 64, 128),   # 28:
#     (192, 64, 128),  # 29:
#     (64, 192, 128),  # 30:
#     (192, 192, 128), # 31:
#     (0, 0, 64),      # 32:
#     (128, 0, 64),    # 33:
#     (0, 128, 64),    # 34:
#     (128, 128, 64),  # 35:
#     (0, 0, 192),     # 36:
#     (128, 0, 192),   # 37:
#     (0, 128, 192),   # 38:
#     (128, 128, 192), # 39:
#     (64, 0, 64),     # 40:
#     (192, 0, 64),    # 41:
# ]

# # --- 코드 본문 (수정 필요 없음) ---

# def visualize_masks():
#     """
#     흑백 마스크 이미지를 컬러 마스크 이미지로 변환하여 시각화합니다.
#     """
#     os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)
    
#     png_files = [f for f in os.listdir(PNG_MASK_DIR) if f.endswith('.png')]
#     print(f"총 {len(png_files)}개의 .png 마스크를 시각화합니다.")

#     processed_count = 0
#     for png_filename in png_files:
#         png_path = os.path.join(PNG_MASK_DIR, png_filename)
        
#         # 흑백 마스크 이미지 로드
#         mask = cv2.imread(png_path, cv2.IMREAD_UNCHANGED) # 1채널 흑백 이미지로 로드
        
#         if mask is None:
#             print(f"경고: 마스크 파일 {png_path}를 읽을 수 없습니다. 건너뜁니다.")
#             continue
            
#         # 컬러 마스크를 저장할 빈 RGB 이미지 생성
#         height, width = mask.shape
#         colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
        
#         # 각 픽셀의 클래스 ID에 따라 색상 적용
#         for class_id in range(len(COLOR_PALETTE)):
#             # 해당 클래스 ID를 가진 모든 픽셀을 찾음
#             # boolean 마스크를 생성하여 해당 픽셀 위치를 식별
#             indices = (mask == class_id)
            
#             # 해당 픽셀 위치에 컬러 팔레트의 색상을 적용
#             # colored_mask[indices] = COLOR_PALETTE[class_id]는 직접적인 assignment가 어려움
#             # 따라서 R, G, B 채널별로 인덱싱하여 값을 할당해야 함
#             colored_mask[indices, 0] = COLOR_PALETTE[class_id][0] # Blue 채널
#             colored_mask[indices, 1] = COLOR_PALETTE[class_id][1] # Green 채널
#             colored_mask[indices, 2] = COLOR_PALETTE[class_id][2] # Red 채널
            
#         # 시각화된 컬러 마스크 저장
#         output_path = os.path.join(OUTPUT_VIS_DIR, png_filename)
#         cv2.imshow("visualize", colored_mask)
#         cv2.waitKey()
#         #cv2.imwrite(output_path, colored_mask)
        
#         processed_count += 1
#         if processed_count % 100 == 0:
#             print(f"{processed_count}/{len(png_files)} 파일 시각화 완료...")

#     print(f"\n시각화 완료! 총 {processed_count}개의 파일을 {OUTPUT_VIS_DIR} 폴더에 저장했습니다.")


# if __name__ == '__main__':
#     visualize_masks()

# 이미지  +  마스크 비교
import cv2
import os

def blend_images_from_folders(base_path):
    image_base = os.path.join(base_path, 'image', 'train')
    colormap_base = os.path.join(base_path, 'colormap', 'train')

    # 두 폴더가 모두 존재하는지 확인
    if not os.path.isdir(image_base) or not os.path.isdir(colormap_base):
        print("오류: 'image/train' 또는 'colormap/train' 폴더를 찾을 수 없습니다. 경로를 확인해주세요.")
        return

    # 'image/train' 폴더의 하위 폴더 목록 가져오기
    subfolders = [d for d in os.listdir(image_base) if os.path.isdir(os.path.join(image_base, d))]
    
    if not subfolders:
        print("경고: 'image/train' 폴더 내부에 하위 폴더가 없습니다.")
        return

    # 하위 폴더들을 순회
    for subfolder in subfolders:
        image_folder = os.path.join(image_base, subfolder)
        colormap_folder = os.path.join(colormap_base, subfolder)

        # image 폴더와 colormap 폴더가 모두 존재하는지 확인
        if not os.path.isdir(colormap_folder):
            print(f"경고: '{colormap_folder}' 폴더가 없어 스킵합니다.")
            continue
        
        # 각 폴더의 파일 목록을 가져와 파일 이름으로 매칭
        # 이미지와 마스크 모두 컬러로 읽기 위해 확장자만 확인
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(image_extensions)])
        colormap_files = sorted([f for f in os.listdir(colormap_folder) if f.lower().endswith(image_extensions)])
        
        # 파일 쌍을 찾아 순회
        for img_name in image_files:
            if img_name in colormap_files:
                img_path = os.path.join(image_folder, img_name)
                colormap_path = os.path.join(colormap_folder, img_name)
                
                # 원본 이미지와 컬러 마스크를 모두 컬러로 읽기
                image = cv2.imread(img_path, cv2.IMREAD_COLOR)
                colormap = cv2.imread(colormap_path, cv2.IMREAD_COLOR)

                if image is None or colormap is None:
                    print(f"오류: '{img_name}' 파일을 읽는 데 실패했습니다. 스킵합니다.")
                    continue
                
                # 이미지와 마스크의 크기 일치 여부 확인
                if image.shape != colormap.shape:
                    print(f"경고: '{img_name}'의 이미지와 마스크 크기가 다릅니다. 스킵합니다.")
                    continue
                
                # cv2.addWeighted()를 사용하여 두 이미지를 겹치기
                # alpha=0.5, beta=0.5로 설정하여 50:50 비율로 합성
                # 두 이미지가 섞여서 결과가 반투명하게 겹쳐 보입니다.
                blended_result = cv2.addWeighted(image, 0.5, colormap, 0.5, 0)
                
                # 최종 결과만 화면에 표시
                cv2.imshow(f'Result: {subfolder}/{img_name}', blended_result)
                
                # 사용자가 아무 키나 누를 때까지 대기 (다음 이미지로 넘어감)
                cv2.waitKey(0)

    # 모든 처리가 끝나면 창 닫기
    cv2.destroyAllWindows()


base_dir = 'C:/ETRI/data/'

blend_images_from_folders(base_dir)