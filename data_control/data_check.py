import os

def count_files_recursively(directory):
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

    for data_type in data_types:
        print(f"\n[검사 대상: {data_type}]")
        
        for split in splits:
            target_dir = os.path.join(base_dir, data_type, split)
            
            if os.path.isdir(target_dir):
                num_files = count_files_recursively(target_dir)
                if num_files > 0:
                    print(f"  ✅ {split:<5s} -> 정상. 총 {num_files}개의 파일이 있습니다.")
                else:
                    print(f"  ⚠️  {split:<5s} -> 경고. 폴더는 존재하지만 비어있습니다.")
                    all_ok = False
            else:
                print(f"  ❌ {split:<5s} -> 오류. 폴더가 존재하지 않습니다.")
                all_ok = False

if __name__ == '__main__':
    dataset_root_path = 'C:\\ETRI\\data'
    

    verify_dataset_structure(dataset_root_path)
