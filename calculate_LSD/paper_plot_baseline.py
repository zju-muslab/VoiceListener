from h import *
import pandas as pd
import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
import matplotlib
NFFT_S = 10
NOVERLAT_S = 15
SHAPE = 129
low, high = 12,25
SR = 500
NEW_SR = 2000

BAND = 5# mag 20 acc 80
MIN_SUM =  4 # mag 3 acc 0.0015 0.0022 gyr 0.00004
SENSORNAME = 'mag'
plt.rcParams["font.family"] = "Times New Roman"

ISTIMIT = 1
# TT = 2

dir = r'E:\MAG\code_final_version\data\huawei_table_lab\aligned\mag'
# dir = r'E:\MAG\code_final_version\data\test_subject\aligned\mag'
audio_dir = r'E:\MAG\code_final_version\data\huawei_table_lab\voice'
# audio_dir = r'E:\MAG\code_final_version\data\test_subject\voice'
def voice_mat(filename):
    # 读取audio
    sampling_rate, audio = read_wav(filename)
    # 过滤高于1000Hz的谐波
    b, a = signal.butter(8, 2 * 1000 / sampling_rate, 'lowpass')  # 配置滤波器 8 表示滤波器的阶数
    audio = signal.filtfilt(b, a, audio)  # data为要过滤的信号
    # 降采样到2000Hz
    audio = audio.copy()[::sampling_rate//2000]
    length = len(audio) // 2000
    audio = audio.copy()[:length*2000]
    # 计算幅度谱
    f, t, Zxx = signal.stft(audio, fs=NEW_SR, nperseg=int(NEW_SR // NFFT_S), noverlap=int(NEW_SR // NOVERLAT_S))
    mat = get_magnitude_mat(Zxx)
    mat /= np.max(mat)

    return mat, length

def voice_recovering_mat(data):

    b, a = signal.butter(8, 2 * BAND / 500, 'highpass')  # 配置滤波器 8 表示滤波器的阶数
    data = signal.filtfilt(b, a, data)  # data为要过滤的信号
    #
    # 恢复幅度谱
    data = add_zero(data)
    data = add_zero(data)

    f, t, Zxx = signal.stft(data, fs=sampling_rate, nfft=128 * 2, nperseg=128 , noverlap=120, padded=False, boundary=None)
    # f, t, Zxx = signal.stft(tmp_audio, fs=sampling_rate, nfft=128*2*2, nperseg=128*2 , noverlap=120*2 ,padded=False, boundary=None)  # 对应AccelEve的audio2img中的specgram函数
    mat = get_magnitude_mat(Zxx)

    p = get_base_freq_modified(mat, low, high, MIN_SUM)
    new_mat= get_new_mat(mat, p)

    if np.max(new_mat)>1e-5:
        new_mat /= np.max(new_mat)

    # 标出基频
    # print(len(p))
    # for j in range(len(p)):
    #     if p[j]!=0:
    #         new_mat[p[j]][j] = 100

    return new_mat


def voice_fold_mat(data):
    # 读取json

    # data = data[:length*500]
    # 过滤直流信号
    b, a = signal.butter(8, 2 * BAND/ 500, 'highpass')  # 配置滤波器 8 表示滤波器的阶数
    data = signal.filtfilt(b, a, data)  # data为要过滤的信号
    # 恢复幅度谱
    data = add_zero(data)
    data = add_zero(data)
    f, t, Zxx = signal.stft(data, fs=sampling_rate, nfft=128 * 2, nperseg=128 , noverlap=120, padded=False, boundary=None)
    mat = get_magnitude_mat(Zxx)
    if np.max(mat)>1e-5:
        mat /= np.max(mat)
    return mat


def voice_translation_mat(data):
    # 读取json
    # data = data[:length * 500]
    # 过滤直流信号
    b, a = signal.butter(8, 2 * BAND / 500, 'highpass')  # 配置滤波器 8 表示滤波器的阶数
    data = signal.filtfilt(b, a, data)  # data为要过滤的信号
    # 恢复幅度谱
    data = spectrum_transaltion(data)
    data = spectrum_transaltion(data)
    f, t, Zxx = signal.stft(data, fs=sampling_rate, nfft=128 * 2, nperseg=128 , noverlap=120, padded=False, boundary=None)
    mat = get_magnitude_mat(Zxx)
    if np.max(mat)>1e-5:
        mat /= np.max(mat)
    return mat


def get_log_power_mat(magnitude_mat):
    return np.log(magnitude_mat*magnitude_mat)

def calculate_distance(mat,recover_mat,raw_mat):

    # mat = get_log_power_mat(mat)
    # recover_mat = get_log_power_mat(recover_mat)
    print("!!!!time",int(0.004*mat.shape[1]))
    print(mat.shape)
    SHAPE1,SHAPE2 = 3611,4361#1361, 2861# 4361#4650
    mat = np.log(mat)[:,SHAPE1:SHAPE2]
    recover_mat = np.log(recover_mat)[:,SHAPE1:SHAPE2]
    raw_mat = np.log(raw_mat)[:,SHAPE1:SHAPE2]

    print(mat.shape)
    d_mat = mat-recover_mat
    k, l = d_mat.shape
    d_mat = d_mat*d_mat
    d_sum = np.sum(d_mat, axis=0)
    d_sum = np.sqrt(d_sum/k)
    result = np.sum(d_sum)/l

    # mat = mat[:,:11050]
    # recover_mat = recover_mat[:, :11050]
    # mat = mat[:, :recover_mat.shape[1]]
    # if mat.shape[1]<10:
    #     return 0
    if 1:
        print("LSD",result)
        plt.figure()
        plt.subplot(2,1,1)
        plt.pcolormesh(np.linspace(0, mat.shape[1], mat.shape[1]), np.linspace(0, mat.shape[0], mat.shape[0]),mat,shading='auto', cmap="magma")
        plt.colorbar()
        plt.subplot(2,1,2)
        plt.colorbar()
        plt.pcolormesh(np.linspace(0, recover_mat .shape[1],recover_mat .shape[1]), np.linspace(0, recover_mat .shape[0], recover_mat .shape[0]),recover_mat,shading='auto', cmap="magma")
        plt.show()
        fff_mat = np.vstack((mat,recover_mat, raw_mat))
        plt.figure()
        plt.pcolormesh(np.linspace(0, fff_mat.shape[1], fff_mat.shape[1]),
                       np.linspace(0, fff_mat.shape[0], fff_mat.shape[0]), fff_mat, shading='auto', cmap="magma")
        print("is_save")

        plt.show()

        font = {'family': 'Times New Roman',
                'color': 'darkred',
                'weight': 'normal',
                'size': 17,
                }

        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42


        mat = mat[:,113:mat.shape[1]]

        raw_mat = raw_mat[:,113:raw_mat.shape[1]]
        recover_mat = recover_mat[:,113:recover_mat.shape[1]]

        plt.figure(figsize=(6,6))
        plt.xlabel("Time (s)", fontsize=20)
        plt.ylabel("Frequency (Hz)", fontsize=18)
        plt.xticks(np.linspace(0, mat.shape[1], 7), ["%.1f" % x for x in np.linspace(0, 0.004 * mat.shape[1], 7)],
                   fontsize=18)
        plt.yticks(fontsize=18)
        plt.subplots_adjust(top=0.88, bottom=0.335, right=0.88, left=0.165, hspace=0, wspace=0)

        plt.pcolormesh(np.linspace(0, mat.shape[1], mat.shape[1]), np.linspace(0, 1000, mat.shape[0]), mat,
                       shading='auto', cmap="magma")
        cb = plt.colorbar(fraction=0.10)# 调整colorbar宽度
        cb.ax.tick_params(labelsize=17)  # 设置色标刻度字体大小。
        cb.set_label('')  # 设置colorbar的标签字体及其大小
        # plt.show()
        plt.savefig(r"d:/evaluation_mat.pdf")





        plt.figure(figsize=(6,6))
        plt.xlabel("Time (s)", fontsize=20)
        plt.ylabel("Frequency (Hz)", fontsize=18)
        plt.xticks(np.linspace(0, mat.shape[1], 7), ["%.1f" % x for x in np.linspace(0, 0.004 * mat.shape[1], 7)],
                   fontsize=18)
        plt.yticks(fontsize=18)
        plt.subplots_adjust(top=0.88, bottom=0.335, right=0.88, left=0.165, hspace=0, wspace=0)

        plt.pcolormesh(np.linspace(0, mat.shape[1], mat.shape[1]), np.linspace(0, 1000, mat.shape[0]), raw_mat,
                       shading='auto', cmap="magma")
        cb = plt.colorbar(fraction=0.10)# 调整colorbar宽度
        cb.ax.tick_params(labelsize=17)  # 设置色标刻度字体大小。
        cb.set_label('')  # 设置colorbar的标签字体及其大小
        plt.savefig(r"d:/evaluation_raw.pdf")


        plt.figure(figsize=(6,6))
        plt.xlabel("Time (s)", fontsize=20)
        plt.ylabel("Frequency (Hz)", fontsize=18)
        plt.xticks(np.linspace(0, mat.shape[1], 7), ["%.1f" % x for x in np.linspace(0, 0.004 * mat.shape[1], 7)],
                   fontsize=18)
        plt.yticks(fontsize=18)
        plt.subplots_adjust(top=0.88, bottom=0.335, right=0.88, left=0.165, hspace=0, wspace=0)

        plt.pcolormesh(np.linspace(0, recover_mat.shape[1], recover_mat.shape[1]), np.linspace(0, 1000, recover_mat.shape[0]), recover_mat,
                       shading='auto', cmap="magma")
        cb = plt.colorbar(fraction=0.10)# 调整colorbar宽度
        cb.ax.tick_params(labelsize=17)  # 设置色标刻度字体大小。
        cb.set_label('')  # 设置colorbar的标签字体及其大小
        # plt.show()
        plt.savefig(r"d:/evaluation_r_mat.pdf")


        # print(r_mat)
        # m,n = mat.shape
        # for i in range(n):
        #     j=0
        #     while j<m and r_mat[j][i]<=1e-2:
        #         j += 1
        #     if j<m//2:
        #         raw_mat[j][i]=2
        #         raw_mat[j][i-1] = 2
        #         raw_mat[j][i+1] = 2
        #         raw_mat[j][i - 2] = 2
        #         raw_mat[j][i + 2] = 2
        #
        # plt.figure(figsize=(12, 2))
        # plt.xlabel("Time (s)", fontsize=20)
        # plt.ylabel("Frequency (Hz)", fontsize=18)
        # plt.xticks(np.linspace(0, mat.shape[1], 7), ["%.1f" % x for x in np.linspace(0, 0.004 * mat.shape[1], 7)],
        #            fontsize=18)
        # plt.yticks(fontsize=18)
        # plt.subplots_adjust(top=0.88, bottom=0.335, right=0.88, left=0.10, hspace=0, wspace=0)
        #
        # plt.pcolormesh(np.linspace(0, mat.shape[1], mat.shape[1]), np.linspace(0, 1000, mat.shape[0]), raw_mat,
        #                shading='auto', cmap="magma")
        # cb = plt.colorbar(aspect=5)
        # cb.ax.tick_params(labelsize=17)  # 设置色标刻度字体大小。
        # cb.set_label('')  # 设置colorbar的标签字体及其大小
        # # plt.show()
        # plt.savefig(r"d:/e_baseline_from_raw.pdf")

    return result


# ids = [3]
f_sum = 0
t_sum = 0
r_sum = 0
ff = []
tt = []
rr = []

true_audio_mat_list = []
f_mat_list = []
t_mat_list = []
r_mat_list = []
raw_mat_list = []

for num_id in range(7,8):
    csv_filename = os.path.join(dir,str(num_id)+'.csv')
    audio_filename = os.path.join(audio_dir,"%02d.wav"%num_id)
    print(csv_filename,audio_filename)
    sensor_data = np.array(pd.read_csv(csv_filename))[:, 2]
    sampling_rate, audio = read_wav(audio_filename)


    sensor_data = sensor_data[:int(len(audio)/sampling_rate*500)]


    num_r_mat_list = []
    num_mat_list = []
    MAX_IMG = (len(audio)//55200)
    # if num_id==3:
    #     MAX_IMG=361
    # for i in tqdm(range(len(audio)//55200)):
    # for i in tqdm(range(int(MAX_IMG*0.8),MAX_IMG)):
    for i in tqdm(range(int(MAX_IMG * 0.2),int(MAX_IMG * 0.25))):
        # wav
        begin = int(i*55200/ISTIMIT)
        end = int((i+1)*55200/ISTIMIT)

        tmp_audio = audio.copy()[begin:end]


        f, t, Zxx = signal.stft(tmp_audio, fs=sampling_rate, nfft=128*48//ISTIMIT,nperseg=128*24//ISTIMIT, noverlap=120*24//ISTIMIT, padded=False,boundary=None) #对应AccelEve的audio2img中的specgram函数
        print("time!!!!!", t[2]-t[1],t[3]-t[2],t[4]-t[3])
        # f, t, Zxx = signal.stft(tmp_audio, fs=48000, nperseg=int(48000 // NFFT_S), noverlap=int(48000 // NOVERLAT_S), padded=False, boundary=None)
        tmp_mat = get_magnitude_mat(Zxx)

        tmp_mat = tmp_mat[:SHAPE:]

        if np.max(tmp_mat) > 1e-5:
            tmp_mat /= np.max(tmp_mat)

        true_audio_mat_list.append(tmp_mat)

        # sensors
        s_begin = int(i*575)
        s_end = int((i+1)*575)
        #
        tmp_sensor_data = sensor_data[s_begin:s_end]


        tmp_r_mat= voice_recovering_mat(tmp_sensor_data)

        b, a = signal.butter(8, 2 * BAND / 500, 'highpass')  # 配置滤波器 8 表示滤波器的阶数
        data = signal.filtfilt(b, a, tmp_sensor_data)  # data为要过滤的信号
        f, t, Zxx = signal.stft(data, fs=sampling_rate, nfft=128 * 2//4, nperseg=128//4, noverlap=120//4, padded=False,
                                boundary=None)
        tmp_raw_mat = get_magnitude_mat(Zxx)
        tmp_raw_mat/=np.max(tmp_raw_mat)
        tmp_raw_mat = np.vstack((tmp_raw_mat,np.zeros((tmp_mat.shape[0]-tmp_raw_mat.shape[0],tmp_raw_mat.shape[1]))))



        tmp_r_mat = np.where(tmp_r_mat < 1e-2, 1e-2, tmp_r_mat)
        tmp_mat = np.where(tmp_mat < 1e-2, 1e-2, tmp_mat)
        tmp_raw_mat = np.where(tmp_raw_mat < 1e-2, 1e-2, tmp_raw_mat)
        raw_mat_list.append(tmp_raw_mat)
        # tmp_lsd=calculate_distance(tmp_r_mat, tmp_mat)
        # lsd_lisd.append(tmp_lsd)
        r_mat_list.append(tmp_r_mat)
        num_r_mat_list.append(tmp_r_mat)
        num_mat_list.append(tmp_mat)
        # plt.figure()
        # plt.subplot(3,1,1)
        # plt.pcolormesh(np.linspace(0, tmp_mat.shape[1], tmp_mat.shape[1]), np.linspace(0, tmp_mat.shape[0], tmp_mat.shape[0]),np.log(tmp_mat),shading='auto', cmap="magma")
        # plt.colorbar()
        # plt.subplot(3,1,2)
        # plt.colorbar()
        # plt.pcolormesh(np.linspace(0, tmp_r_mat .shape[1],tmp_r_mat .shape[1]), np.linspace(0, tmp_r_mat .shape[0], tmp_r_mat .shape[0]),np.log(tmp_r_mat) ,shading='auto', cmap="magma")
        # plt.subplot(3, 1,3)
        # plt.plot(np.sum(tmp_mat, 1))
        # plt.show()

    num_r_mat= np.hstack([i for i in num_r_mat_list])
    num_mat = np.hstack([i for i in num_mat_list])
    # print(num_id, calculate_distance(num_mat, num_r_mat))


true_audio_mat = np.hstack([i for i in true_audio_mat_list])
true_audio_mat = true_audio_mat[:SHAPE,:]
mat = np.where(true_audio_mat < 1e-2, 1e-2, true_audio_mat)
r_mat = np.hstack([i for i in r_mat_list])
raw_mat = np.hstack([i for i in raw_mat_list])
# audio_filename = os.path.join(audio_dir,"%02d.wav"%1)
# sampling_rate, audio = read_wav(audio_filename)
# f, t, Zxx = signal.stft(audio, fs=sampling_rate, nfft=128*48//ISTIMIT,nperseg=128*24//ISTIMIT, noverlap=120*24//ISTIMIT, padded=False,boundary=None)
# r_mat = get_magnitude_mat(Zxx)[:,:mat.shape[1]]
# r_mat /= r_mat.max()
# f_mat = voice_fold_mat(sensor_data)
# t_mat = voice_translation_mat(sensor_data)
# r_mat = voice_recovering_mat(sensor_data)


# 预处理
mat = np.where(mat < 1e-2, 1e-2, mat)
r_mat = np.where(r_mat < 1e-2, 1e-2, r_mat)


d3 = calculate_distance(mat, r_mat,raw_mat)



# print( "our method ", d3)
