# 훈련, 검증, 테스트 7:2:1 나누는  코드

# import os
# import shutil
# from pathlib import Path
# from typing import Union, List

# def split_dataset(base_dir):
#     """
#     모든 폴더의 다양한 파일명 규칙을 모두 지원하여 데이터 분할을 수행합니다.
#     """
#     # ====== [수정] 폴더/파일명 패턴 설정 ======
#     main_folders = ['colormap', 'image', 'labelmap']

#     # [수정] 더 구체적인 규칙(_leftImg8bit.png)을 일반적인 규칙(.png)보다 앞에 배치합니다.
#     IMAGE_SUFFIXES = ["_leftImg8bit.png", ".jpg", ".png"] 

#     # 라벨/컬러맵 접미사 후보 리스트 (이전과 동일하게 유지)
#     LABELMAP_SUFFIXES = ["_gtFine_CategoryId.png", "_CategoryId.png"]
#     COLORMAP_SUFFIXES = ["_gtFine_Color.png", "._color.png", "_color.png"]
    
#     def map_filename_for_folder(main_folder: str, img_filename: str, src_dir: Path) -> Union[str, None]:
#         base_name = None
#         for img_suffix in IMAGE_SUFFIXES:
#             if img_filename.endswith(img_suffix):
#                 base_name = img_filename[:-len(img_suffix)]
#                 break
#         if base_name is None:
#             return None
#         if main_folder == 'image':
#             return img_filename
        
#         suffix_candidates: List[str] = []
#         if main_folder == 'labelmap':
#             suffix_candidates = LABELMAP_SUFFIXES
#         elif main_folder == 'colormap':
#             suffix_candidates = COLORMAP_SUFFIXES

#         for suffix in suffix_candidates:
#             candidate_name = base_name + suffix
#             if (src_dir / candidate_name).exists():
#                 return candidate_name
#         return None

#     source_base_path = Path(base_dir) / 'image' / 'train'
#     if not source_base_path.exists():
#         print(f"오류: 기준 경로 '{source_base_path}'를 찾을 수 없습니다.")
#         return

#     try:
#         sub_folders = [d.name for d in source_base_path.iterdir() if d.is_dir()]
#     except OSError as e:
#         print(f"오류: '{source_base_path}' 폴더를 읽는 중 문제가 발생했습니다: {e}")
#         return

#     print("모든 하위 폴더 및 파일 규칙을 대상으로 데이터 분할을 시작합니다...")

#     for sub_folder in sub_folders:
#         print(f"\n📁 [{sub_folder}] 폴더 처리 중...")
#         source_sub_folder_path = source_base_path / sub_folder
#         try:
#             files = sorted([f.name for f in source_sub_folder_path.iterdir() if f.is_file()])
#         except FileNotFoundError:
#             print(f"  '{source_sub_folder_path}' 폴더를 찾을 수 없어 건너뜁니다.")
#             continue
#         if not files:
#             print(f"  '{sub_folder}' 폴더에 파일이 없어 건너뜁니다.")
#             continue

#         print(f"  (1/2) 사전 검증을 시작합니다...")
#         move_plan = []
#         is_plan_valid = True
#         temp_moved_counts = {'train': 0, 'val': 0, 'test': 0}
        
#         for i in range(0, len(files), 10):
#             chunk = files[i:i+10]
#             files_to_move_map = {}
#             if len(chunk) == 10:
#                 files_to_move_map = {'val': chunk[7:9], 'test': chunk[9:10]}
            
#             train_count = len(chunk) - len(files_to_move_map.get('val', [])) - len(files_to_move_map.get('test', []))
#             temp_moved_counts['train'] += train_count

#             for split_type, files_to_move in files_to_move_map.items():
#                 if not files_to_move: continue
#                 for img_name in files_to_move:
#                     temp_moved_counts[split_type] += 1
#                     for main_folder in main_folders:
#                         src_dir = Path(base_dir) / main_folder / 'train' / sub_folder
#                         mapped_name = map_filename_for_folder(main_folder, img_name, src_dir)
#                         if mapped_name:
#                             source_file = src_dir / mapped_name
#                             dest_file = Path(base_dir) / main_folder / split_type / sub_folder / mapped_name
#                             move_plan.append({'src': source_file, 'dest': dest_file})
#                         else:
#                             if main_folder != 'image':
#                                 print(f"  [검증 실패] 원본 '{img_name}'에 해당하는 '{main_folder}' 파일을 찾을 수 없습니다.")
#                                 print(f"  -> '{sub_folder}' 폴더의 모든 파일 이동 작업을 취소합니다.")
#                                 is_plan_valid = False
#                                 break
#                     if not is_plan_valid: break
#                 if not is_plan_valid: break
#             if not is_plan_valid: break

#         if is_plan_valid:
#             print(f"  (2/2) 검증 완료. 총 {len(move_plan)}개 파일 이동을 시작합니다.")
#             for move_op in move_plan:
#                 move_op['dest'].parent.mkdir(parents=True, exist_ok=True)
#                 shutil.move(str(move_op['src']), str(move_op['dest']))
            
#             print(f"  - ✅ Train: {temp_moved_counts['train']}개 파일 유지")
#             print(f"  - ✅ Validation: {temp_moved_counts['val']}개 파일 이동 완료")
#             print(f"  - ✅ Test: {temp_moved_counts['test']}개 파일 이동 완료")
#         else:
#             print(f"  (2/2) 작업이 취소되어 [{sub_folder}] 폴더의 파일이 이동되지 않았습니다.")

#     print("\n🎉 모든 작업이 완료되었습니다.")

# # --- 실행 예시 ---
# if __name__ == '__main__':
#     base_directory = r'C:/ETRI/data'
#     split_dataset(base_directory)


# 훈련 검증 8:2
import os
import shutil
from pathlib import Path
from typing import Union, List

def split_dataset(base_dir):
    """
    모든 폴더의 다양한 파일명 규칙을 지원하여 8:2 비율로 train/val 데이터 분할을 수행합니다.
    (10개 묶음 중 8개는 train, 2개는 val로 분할)
    """
    # ====== [설정] 폴더/파일명 패턴 설정 ======
    main_folders = ['colormap', 'image', 'labelmap']

    IMAGE_SUFFIXES = ["_leftImg8bit.png", ".jpg", ".png"] 
    LABELMAP_SUFFIXES = ["_gtFine_CategoryId.png", "_CategoryId.png"]
    COLORMAP_SUFFIXES = ["_gtFine_Color.png", "._color.png", "_color.png"]
    
    def map_filename_for_folder(main_folder: str, img_filename: str, src_dir: Path) -> Union[str, None]:
        base_name = None
        for img_suffix in IMAGE_SUFFIXES:
            if img_filename.endswith(img_suffix):
                base_name = img_filename[:-len(img_suffix)]
                break
        if base_name is None:
            return None
        if main_folder == 'image':
            return img_filename
        
        suffix_candidates: List[str] = []
        if main_folder == 'labelmap':
            suffix_candidates = LABELMAP_SUFFIXES
        elif main_folder == 'colormap':
            suffix_candidates = COLORMAP_SUFFIXES

        for suffix in suffix_candidates:
            candidate_name = base_name + suffix
            if (src_dir / candidate_name).exists():
                return candidate_name
        return None

    source_base_path = Path(base_dir) / 'image' / 'train'
    if not source_base_path.exists():
        print(f"오류: 기준 경로 '{source_base_path}'를 찾을 수 없습니다.")
        return

    try:
        sub_folders = [d.name for d in source_base_path.iterdir() if d.is_dir()]
    except OSError as e:
        print(f"오류: '{source_base_path}' 폴더를 읽는 중 문제가 발생했습니다: {e}")
        return

    print("모든 하위 폴더 및 파일 규칙을 대상으로 8:2 (train:val) 분할을 시작합니다...")

    for sub_folder in sub_folders:
        print(f"\n📁 [{sub_folder}] 폴더 처리 중...")
        source_sub_folder_path = source_base_path / sub_folder
        try:
            files = sorted([f.name for f in source_sub_folder_path.iterdir() if f.is_file()])
        except FileNotFoundError:
            print(f"  '{source_sub_folder_path}' 폴더를 찾을 수 없어 건너뜁니다.")
            continue
        if not files:
            print(f"  '{sub_folder}' 폴더에 파일이 없어 건너뜁니다.")
            continue

        print(f"  (1/2) 사전 검증을 시작합니다...")
        move_plan = []
        is_plan_valid = True
        
        # [수정] 'test' 키 제거
        temp_moved_counts = {'train': 0, 'val': 0}
        
        for i in range(0, len(files), 10):
            chunk = files[i:i+10]
            files_to_move_map = {}
            if len(chunk) == 10:
                # [수정] 8:2 비율로 변경. 8개는 train(유지), 2개(8번째, 9번째)는 val(이동)
                files_to_move_map = {'val': chunk[8:10]}
            
            # [수정] 'test' 관련 계산 제거
            train_count = len(chunk) - len(files_to_move_map.get('val', []))
            temp_moved_counts['train'] += train_count

            # 'test'가 files_to_move_map에 없으므로 이 루프는 'val'만 처리하게 됨
            for split_type, files_to_move in files_to_move_map.items(): 
                if not files_to_move: continue
                for img_name in files_to_move:
                    temp_moved_counts[split_type] += 1
                    for main_folder in main_folders:
                        src_dir = Path(base_dir) / main_folder / 'train' / sub_folder
                        mapped_name = map_filename_for_folder(main_folder, img_name, src_dir)
                        if mapped_name:
                            source_file = src_dir / mapped_name
                            dest_file = Path(base_dir) / main_folder / split_type / sub_folder / mapped_name
                            move_plan.append({'src': source_file, 'dest': dest_file})
                        else:
                            if main_folder != 'image':
                                print(f"  [검증 실패] 원본 '{img_name}'에 해당하는 '{main_folder}' 파일을 찾을 수 없습니다.")
                                print(f"  -> '{sub_folder}' 폴더의 모든 파일 이동 작업을 취소합니다.")
                                is_plan_valid = False
                                break
                    if not is_plan_valid: break
                if not is_plan_valid: break
            if not is_plan_valid: break

        if is_plan_valid:
            print(f"  (2/2) 검증 완료. 총 {len(move_plan)}개 파일 이동을 시작합니다.")
            for move_op in move_plan:
                move_op['dest'].parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(move_op['src']), str(move_op['dest']))
            
            print(f"  - ✅ Train: {temp_moved_counts['train']}개 파일 유지")
            print(f"  - ✅ Validation: {temp_moved_counts['val']}개 파일 이동 완료")
            # [수정] 'test' 출력 라인 제거
        else:
            print(f"  (2/2) 작업이 취소되어 [{sub_folder}] 폴더의 파일이 이동되지 않았습니다.")

    print("\n🎉 모든 작업이 완료되었습니다.")

# --- 실행 예시 ---
if __name__ == '__main__':
    base_directory = r'C:/ETRI/data'
    split_dataset(base_directory)