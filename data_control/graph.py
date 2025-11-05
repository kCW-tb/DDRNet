import matplotlib.pyplot as plt
import pandas as pd
import os
import io

def plot_and_save_metrics_from_log(save_dir="training_plots"):
    # Epoch, Train-loss, Val-loss, learningRate 순서
    log_data = """
        Epoch		Train-loss		Val-loss		learningRate

        1		5.9139		N/A		0.00002004
        2		3.6741		N/A		0.00004003
        3		2.6756		N/A		0.00006002
        4		2.3517		N/A		0.00008001
        5		2.1469		1.3198		0.00010000
        6		1.9777		N/A		0.00009969
        7		1.7760		N/A		0.00009939
        8		1.6182		N/A		0.00009908
        9		1.5070		N/A		0.00009878
        10		1.4134		0.8850		0.00009847
        11		1.3493		N/A		0.00009817
        12		1.2931		N/A		0.00009786
        13		1.2440		N/A		0.00009756
        14		1.1871		N/A		0.00009725
        15		1.1532		0.6542		0.00009694
        16		1.1240		N/A		0.00009664
        17		1.0961		N/A		0.00009633
        18		1.0627		N/A		0.00009603
        19		1.0366		N/A		0.00009572
        20		1.0144		0.5590		0.00009541
        21		0.9957		N/A		0.00009511
        22		0.9820		N/A		0.00009480
        23		0.9461		N/A		0.00009449
        24		0.9363		N/A		0.00009418
        25		0.9161		0.5226		0.00009388
        26		0.9116		N/A		0.00009357
        27		0.8932		N/A		0.00009326
        28		0.8987		N/A		0.00009295
        29		0.8770		N/A		0.00009265
        30		0.8670		0.4612		0.00009234
        31		0.8523		N/A		0.00009203
        32		0.8240		N/A		0.00009172
        33		0.8424		N/A		0.00009142
        34		0.8123		N/A		0.00009111
        35		0.8165		0.4332		0.00009080
        36		0.8053		N/A		0.00009049
        37		0.7938		N/A		0.00009018
        38		0.7983		N/A		0.00008987
        39		0.7834		N/A		0.00008956
        40		0.7794		0.4151		0.00008926
        41		0.7606		N/A		0.00008895
        42		0.7613		N/A		0.00008864
        43		0.7576		N/A		0.00008833
        44		0.7568		N/A		0.00008802
        45		0.7411		0.3871		0.00008771
        46		0.7402		N/A		0.00008740
        47		0.7284		N/A		0.00008709
        48		0.7217		N/A		0.00008678
        49		0.7170		N/A		0.00008647
        50		0.7177		0.3749		0.00008616
        51		0.7151		N/A		0.00008585
        52		0.7064		N/A		0.00008554
        53		0.7086		N/A		0.00008523
        54		0.6973		N/A		0.00008492
        55		0.6923		0.3723		0.00008461
        56		0.6972		N/A		0.00008430
        57		0.6805		N/A		0.00008399
        58		0.6824		N/A		0.00008367
        59		0.6743		N/A		0.00008336
        60		0.6706		0.3560		0.00008305
        61		0.6689		N/A		0.00008274
        62		0.6678		N/A		0.00008243
        63		0.6609		N/A		0.00008212
        64		0.6581		N/A		0.00008181
        65		0.6631		0.3545		0.00008149
        66		0.6575		N/A		0.00008118
        67		0.6541		N/A		0.00008087
        68		0.6490		N/A		0.00008056
        69		0.6398		N/A		0.00008024
        70		0.6263		0.3490		0.00007993
        71		0.6354		N/A		0.00007962
        72		0.6427		N/A		0.00007931
        73		0.6282		N/A		0.00007899
        74		0.6402		N/A		0.00007868
        75		0.6276		0.3206		0.00007837
        76		0.6347		N/A		0.00007805
        77		0.6224		N/A		0.00007774
        78		0.6125		N/A		0.00007742
        79		0.6163		N/A		0.00007711
        80		0.6109		0.3301		0.00007680
        81		0.6135		N/A		0.00007648
        82		0.6068		N/A		0.00007617
        83		0.5898		N/A		0.00007585
        84		0.6029		N/A		0.00007554
        85		0.5941		0.3238		0.00007522
        86		0.5955		N/A		0.00007491
        87		0.5923		N/A		0.00007459
        88		0.5867		N/A		0.00007428
        89		0.5880		N/A		0.00007396
        90		0.5872		0.3281		0.00007365
        91		0.5850		N/A		0.00007333
        92		0.5824		N/A		0.00007302
        93		0.5891		N/A		0.00007270
        94		0.5747		N/A		0.00007238
        95		0.5742		0.3097		0.00007207
        96		0.5804		N/A		0.00007175
        97		0.5779		N/A		0.00007143
        98		0.5790		N/A		0.00007112
        99		0.5727		N/A		0.00007080
        100		0.5763		0.3092		0.00007048
    """
    

    df = pd.read_csv(io.StringIO(log_data), delim_whitespace=True, 
                      names=['Epoch', 'Train-loss', 'Val-loss', 'learningRate'], header=0)
    df['Val-loss'] = pd.to_numeric(df['Val-loss'], errors='coerce')
    
    # 저장할 디렉터리가 없으면 생성
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(12, 7))
    plt.plot(df['Epoch'], df['Train-loss'], color='blue', marker='o', linestyle='-', markersize=4, label='Train Loss')
    
    val_df = df.dropna(subset=['Val-loss'])
    plt.plot(val_df['Epoch'], val_df['Val-loss'], color='green', marker='s', linestyle='--', markersize=5, label='Validation Loss')
    
    # 그래프 제목 및 라벨 설정
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # 그래프를 파일로 저장
    loss_path = os.path.join(save_dir, 'loss_curves.png')
    plt.savefig(loss_path)
    plt.close()
    print(f"Loss 그래프가 '{loss_path}'에 저장되었습니다.")

    # --- learningRate 그래프 ---
    plt.figure(figsize=(12, 7))
    plt.plot(df['Epoch'], df['learningRate'], color='red', marker='x', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate over Epochs')
    plt.grid(True)
    plt.tight_layout()
    
    # 그래프를 파일로 저장
    lr_path = os.path.join(save_dir, 'learning_rate.png')
    plt.savefig(lr_path)
    plt.close()
    print(f"Learning rate 그래프가 '{lr_path}'에 저장되었습니다.")

# --- 스크립트 실행 시작점 ---
if __name__ == '__main__':

    plot_and_save_metrics_from_log()
