import cv2
import os

def blend_images_from_folders(base_path):
    image_base = os.path.join(base_path, 'image', 'train')
    colormap_base = os.path.join(base_path, 'colormap', 'train')

    if not os.path.isdir(image_base) or not os.path.isdir(colormap_base):
        print("오류: 'image/train' 또는 'colormap/train' 폴더를 찾을 수 없습니다. 경로를 확인해주세요.")
        return

    subfolders = [d for d in os.listdir(image_base) if os.path.isdir(os.path.join(image_base, d))]
    
    if not subfolders:
        print("경고: 'image/train' 폴더 내부에 하위 폴더가 없습니다.")
        return

    for subfolder in subfolders:
        image_folder = os.path.join(image_base, subfolder)
        colormap_folder = os.path.join(colormap_base, subfolder)

        if not os.path.isdir(colormap_folder):
            print(f"경고: '{colormap_folder}' 폴더가 없어 스킵합니다.")
            continue
        
        # 확장자 확인
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(image_extensions)])
        colormap_files = sorted([f for f in os.listdir(colormap_folder) if f.lower().endswith(image_extensions)])
        
        # 파일 쌍을 찾아 순회
        for img_name in image_files:
            if img_name in colormap_files:
                img_path = os.path.join(image_folder, img_name)
                colormap_path = os.path.join(colormap_folder, img_name)
                
                image = cv2.imread(img_path, cv2.IMREAD_COLOR)
                colormap = cv2.imread(colormap_path, cv2.IMREAD_COLOR)

                if image is None or colormap is None:
                    print(f"오류: '{img_name}' 파일을 읽는 데 실패.")
                    continue
                
                # 이미지와 마스크의 크기 일치 여부 확인
                if image.shape != colormap.shape:
                    print(f"경고: '{img_name}'의 이미지와 마스크 크기 오류.")
                    continue
                
                blended_result = cv2.addWeighted(image, 0.5, colormap, 0.5, 0)
                
                cv2.imshow(f'Result: {subfolder}/{img_name}', blended_result)
                
                cv2.waitKey(0)

    cv2.destroyAllWindows()

base_dir = 'C:/ETRI/data/'


blend_images_from_folders(base_dir)
