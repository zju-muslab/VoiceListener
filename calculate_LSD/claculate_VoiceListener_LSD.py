from h import *
import pandas as pd
import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
NFFT_S = 10
NOVERLAT_S = 15
SHAPE = 129
low, high = 5, 20
SR = 500
NEW_SR = 2000

BAND = 80 # mag 20 acc 80
MIN_SUM =  0.00010 # mag 3 acc 0.0015 0.0022 gyr 0.00004  补充实验用的acc 0.0022   xiaomi的 acc 0.003


ISTIMIT = 1
# TT = 2
MAX_IMG =60# 462
audio_filename = r'E:\MAG\code_final_version\data\robustness\voice\test_robustness.wav'
# audio_filename = r'E:\MAG\code_final_version\data\huawei_table_lab\digit\voice\9.wav'
audio_filename = r'E:\MAG\code_final_version\data\in_the_wild\voice\DR3.wav'
# audio_filename = r'E:\MAG\code_final_version\data\huawei_table_lab\digit\voice\9.wav'
sampling_rate, audio = read_wav(audio_filename)
# read csv
# csv_filename = r'E:\MAG\code_final_version\data\in_the_wild\aligned\mag\Acc.csv'
# csv_filename = r'E:\MAG\code_final_version\data\huawei_table_lab\digit\sensor\acc\9.csv'


# csv_filename = r'E:\MAG\code_final_version\data\robustness\huawei_p20_pro\aligned\acc\Accelerometer.csv'
# csv_filename = r'E:\MAG\code_final_version\data\robustness\huawei_p20_pro\aligned\gyr\Gyroscope.csv'
# csv_filename = r'E:\MAG\code_final_version\data\robustness\huawei_honor_v30\aligned\gyr\Gyroscope.csv'
# csv_filename = r'E:\MAG\code_final_version\data\robustness\motorala_edge_S\aligned\gyr\Gyroscope.csv'
# csv_filename = r'E:\MAG\code_final_version\data\robustness\redmi_k30_pro\aligned\gyr\Gyroscope.csv'
# csv_filename = r'E:\MAG\code_final_version\data\robustness\VIVO_IQOO3\aligned\gyr\Gyroscope.csv'
# csv_filename = r'E:\MAG\code_final_version\data\robustness\samsung_S20_Ultra\aligned\gyr\Gyroscope.csv'
# csv_filename = r'E:\MAG\code_final_version\data\robustness\samsung_S21\aligned\gyr\Gyroscope.csv'
# csv_filename = r'E:\MAG\code_final_version\data\robustness\oppo\aligned\acc\Accelerometer.csv'
# csv_filename = r'E:\MAG\code_final_version\data\robustness\xiaomi\aligned\acc\Accelerometer.csv'


csv_filename = r'E:\MAG\code_final_version\data\in_the_wild\aligned\mag\DR3.csv'
# csv_filename = r'E:\MAG\code_final_version\data\huawei_table_lab\aligned\mag'

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
    new_mat = get_new_mat(mat, p)
    if np.max(new_mat)>1e-5:
        new_mat /= np.max(new_mat)

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

def calculate_distance(mat,recover_mat):
    # points = []
    # mat_sum = np.sum(mat, axis=0)
    # for idx, i in enumerate(mat_sum):
    #     if i >3:
    #         points.append(idx)
    # mat = mat[:,points]
    # recover_mat = recover_mat[:, points]
    # print("points",points)


    mat = get_log_power_mat(mat)
    recover_mat = get_log_power_mat(recover_mat)
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
    print("shape",mat.shape)
    if 1:
        plt.figure()
        plt.subplot(2,1,1)
        plt.pcolormesh(np.linspace(0, mat.shape[1], mat.shape[1]), np.linspace(0, mat.shape[0], mat.shape[0]),mat,shading='auto', cmap="magma")
        plt.colorbar()
        plt.subplot(2,1,2)
        plt.colorbar()
        plt.pcolormesh(np.linspace(0, recover_mat .shape[1],recover_mat .shape[1]), np.linspace(0, recover_mat .shape[0], recover_mat .shape[0]),recover_mat,shading='auto', cmap="magma")
        plt.show()
    return result


# ids = [3]
f_sum = 0
t_sum = 0
r_sum = 0
ff = []
tt = []
rr = []


# sensor data 的时间长度控制成和测试音频时长一样
sensor_data = np.array(pd.read_csv(csv_filename))[:, 2]
sensor_data = sensor_data[:int(len(audio)/sampling_rate*500)]

true_audio_mat_list = []
f_mat_list = []
t_mat_list = []
r_mat_list = []


MAX_IMG = (len(audio) // 55200)
print(MAX_IMG)
for i in tqdm(range(int(MAX_IMG * 0.8), MAX_IMG)):
# for i in tqdm(range(MAX_IMG)):
    # wav
    begin = int(i*55200/ISTIMIT)
    end = int((i+1)*55200/ISTIMIT)

    tmp_audio = audio.copy()[begin:end]


    f, t, Zxx = signal.stft(tmp_audio, fs=sampling_rate, nfft=128*48//ISTIMIT,nperseg=128*24//ISTIMIT, noverlap=120*24//ISTIMIT, padded=False,boundary=None) #对应AccelEve的audio2img中的specgram函数
    # f, t, Zxx = signal.stft(tmp_audio, fs=48000, nperseg=int(48000 // NFFT_S), noverlap=int(48000 // NOVERLAT_S), padded=False, boundary=None)
    tmp_mat = get_magnitude_mat(Zxx)

    tmp_mat = tmp_mat[:SHAPE:]

    if np.max(tmp_mat) > 1e-5:
        tmp_mat /= np.max(tmp_mat)



    # sensors
    s_begin = int(i*575)
    s_end = int((i+1)*575)
    #
    tmp_sensor_data = sensor_data[s_begin:s_end]
    if len(tmp_sensor_data)<s_end-s_begin:
        print("calculate",int(MAX_IMG * 0.8), " to ", i-1)
        break

    tmp_r_mat = voice_recovering_mat(tmp_sensor_data)
    true_audio_mat_list.append(tmp_mat)

    tmp_r_mat = np.where(tmp_r_mat < 1e-2, 1e-2, tmp_r_mat)
    tmp_mat = np.where(tmp_mat < 1e-2, 1e-2, tmp_mat)

    # tmp_lsd=calculate_distance(tmp_r_mat, tmp_mat)
    # lsd_lisd.append(tmp_lsd)
    r_mat_list.append(tmp_r_mat)

    plt.figure()
    plt.subplot(3,1,1)
    plt.pcolormesh(np.linspace(0, tmp_mat.shape[1], tmp_mat.shape[1]), np.linspace(0, tmp_mat.shape[0], tmp_mat.shape[0]),np.log(tmp_mat),shading='auto', cmap="magma")
    plt.colorbar()
    plt.subplot(3,1,2)
    plt.colorbar()
    plt.pcolormesh(np.linspace(0, tmp_r_mat .shape[1],tmp_r_mat .shape[1]), np.linspace(0, tmp_r_mat .shape[0], tmp_r_mat .shape[0]),np.log(tmp_r_mat) ,shading='auto', cmap="magma")
    plt.subplot(3, 1,3)
    plt.plot(np.sum(tmp_mat, 1))
    plt.show()

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(tmp_audio)
    plt.subplot(2, 1, 2)
    plt.plot(tmp_sensor_data)
    plt.show()


true_audio_mat = np.hstack([i for i in true_audio_mat_list])
true_audio_mat = true_audio_mat[:SHAPE,:]
mat = np.where(true_audio_mat < 1e-2, 1e-2, true_audio_mat)




r_mat = np.hstack([i for i in r_mat_list])

# f_mat = voice_fold_mat(sensor_data)
# t_mat = voice_translation_mat(sensor_data)
# r_mat = voice_recovering_mat(sensor_data)


# 预处理
mat = np.where(mat < 1e-2, 1e-2, mat)
r_mat = np.where(r_mat < 1e-2, 1e-2, r_mat)



d3 = calculate_distance(mat, r_mat)


print( "our method ", d3)
