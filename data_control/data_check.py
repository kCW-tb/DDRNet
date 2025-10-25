import os

def count_files_recursively(directory):
    """지정된 디렉토리와 모든 하위 디렉토리의 파일 총 개수를 계산합니다."""
    total_files = 0
    try:
        for root, dirs, files in os.walk(directory):
            total_files += len(files)
    except FileNotFoundError:
        return 0
    return total_files

def verify_dataset_structure(base_dir):
    """
    데이터셋의 기본 구조(image, labelmap, colormap)와
    각각의 train/val 폴더 존재 여부 및 파일 개수를 확인합니다.
    """
    print(f"--- '{base_dir}' 데이터셋 구조 검사를 시작합니다 ---")
    
    data_types = ['image', 'labelmap', 'colormap']
    splits = ['train', 'val']
    
    all_ok = True

    # 1. 각 데이터 타입(image, labelmap, colormap)에 대해 반복
    for data_type in data_types:
        print(f"\n[검사 대상: {data_type}]")
        
        # 2. 각 데이터 타입 내의 train, val 폴더 확인
        for split in splits:
            # 확인할 폴더의 전체 경로 생성
            target_dir = os.path.join(base_dir, data_type, split)
            
            # 3. 폴더 존재 여부 확인
            if os.path.isdir(target_dir):
                # 4. 폴더가 존재하면 내부 파일 개수 확인
                num_files = count_files_recursively(target_dir)
                if num_files > 0:
                    print(f"  ✅ {split:<5s} -> 정상. 총 {num_files}개의 파일이 있습니다.")
                else:
                    print(f"  ⚠️  {split:<5s} -> 경고. 폴더는 존재하지만 비어있습니다.")
                    all_ok = False
            else:
                print(f"  ❌ {split:<5s} -> 오류. 폴더가 존재하지 않습니다.")
                all_ok = False
    
    print("\n-------------------------------------------------")
    if all_ok:
        print("✅ 최종 결과: 모든 폴더와 데이터가 정상적으로 확인되었습니다.")
    else:
        print("❌ 최종 결과: 일부 폴더에 문제가 발견되었습니다. 위의 로그를 확인해주세요.")
    print("-------------------------------------------------")


if __name__ == '__main__':
    # 여기에 실제 데이터셋이 있는 상위 폴더 경로를 입력하세요.
    # 예: 'C:\\ETRI\\data'
    dataset_root_path = 'C:\\ETRI\\data'
    
    verify_dataset_structure(dataset_root_path)