#데이터셋을 train, val, test용 lst파일 생성 코드

import os

def create_lst_files(base_path):
    """
    데이터셋 경로를 기반으로 train, val, test 용 .lst 파일을 생성합니다.
    .lst 파일의 각 줄 형식: "split/images/filename.jpg split/labels/filename.txt"

    :param base_path: 'train', 'val', 'test' 폴더가 있는 기본 경로
    """
    splits = ['train', 'val', 'test']
    print(f"'{base_path}' 경로에서 .lst 파일 생성을 시작합니다.")

    for split in splits:
        # 현재 분할(split)에 대한 이미지 폴더 경로
        # 예: C:\DDRNet\data\ETRI\train\images
        image_dir = os.path.join(base_path, split, 'images')

        # 이미지 폴더가 없으면 건너뜀
        if not os.path.isdir(image_dir):
            print(f"경고: '{image_dir}' 폴더를 찾을 수 없어 건너뜁니다.")
            continue

        # .lst 파일에 쓸 내용을 담을 리스트
        lst_content = []
        
        # 이미지 폴더 내의 모든 파일 목록을 가져옴
        image_files = os.listdir(image_dir)
        
        # jpg 파일만 필터링하고 정렬
        jpg_files = sorted([f for f in image_files if f.lower().endswith('.jpg')])

        # 각 이미지 파일에 대해 .lst 파일 형식에 맞는 문자열 생성
        for image_file in jpg_files:
            # 확장자를 제외한 순수 파일 이름 추출
            base_name = os.path.splitext(image_file)[0]
            
            # .lst 파일에 기록될 상대 경로 생성
            # many DL frameworks prefer forward slashes even on Windows
            image_path_relative = f"{split}/images/{base_name}.jpg"
            label_path_relative = f"{split}/labels/{base_name}.txt"
            
            # 최종 라인 생성: "이미지경로 라벨경로"
            line = f"{image_path_relative} {label_path_relative}"
            lst_content.append(line)

        # .lst 파일 저장
        if lst_content:
            lst_file_path = os.path.join(base_path, f"{split}.lst")
            with open(lst_file_path, 'w') as f:
                f.write('\n'.join(lst_content))
            print(f"✅ '{lst_file_path}' 파일 생성 완료! (총 {len(lst_content)} 줄)")
        else:
            print(f"'{image_dir}' 경로에 처리할 .jpg 파일이 없습니다.")

# --- 스크립트 실행 ---
# 데이터셋 기본 경로를 설정합니다.
dataset_base_path = r'C:\DDRNet\data\ETRI'
create_lst_files(dataset_base_path)