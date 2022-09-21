from h import *
import pandas as pd
import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
NFFT_S = 10
NOVERLAT_S = 15
SHAPE = 129
low, high = 10, 50
SR = 500
NEW_SR = 2000

BAND = 5# mag 20 acc 80
MIN_SUM =  3 # mag 3 acc 0.0015 0.0022 gyr 0.00004
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

true_audio_mat_list = []
f_mat_list = []
t_mat_list = []
r_mat_list = []


for num_id in range(1,11):
    csv_filename = os.path.join(dir,str(num_id)+'.csv')
    audio_filename = os.path.join(audio_dir,"%02d.wav"%num_id)
    print(csv_filename,audio_filename)
    sensor_data = np.array(pd.read_csv(csv_filename))[:, 2]
    sampling_rate, audio = read_wav(audio_filename)

    sensor_data = sensor_data[:int(len(audio)/sampling_rate*500)]


    num_r_mat_list = []
    num_mat_list = []
    MAX_IMG = (len(audio)//55200)
    if num_id==3:
        MAX_IMG=361
    for i in tqdm(range(len(audio)//55200)):
    # for i in tqdm(range(int(MAX_IMG*0.8),MAX_IMG)):
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

        true_audio_mat_list.append(tmp_mat)

        # sensors
        s_begin = int(i*575)
        s_end = int((i+1)*575)
        #
        tmp_sensor_data = sensor_data[s_begin:s_end]

        tmp_f_mat = voice_fold_mat(tmp_sensor_data)
        tmp_t_mat = voice_translation_mat(tmp_sensor_data)
        tmp_r_mat = voice_recovering_mat(tmp_sensor_data)


        tmp_r_mat = np.where(tmp_r_mat < 1e-2, 1e-2, tmp_r_mat)
        tmp_mat = np.where(tmp_mat < 1e-2, 1e-2, tmp_mat)

        # tmp_lsd=calculate_distance(tmp_r_mat, tmp_mat)
        # lsd_lisd.append(tmp_lsd)
        r_mat_list.append(tmp_r_mat)
        f_mat_list.append(tmp_f_mat)
        t_mat_list.append(tmp_t_mat)

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
    print(num_id, calculate_distance(num_mat, num_r_mat))


true_audio_mat = np.hstack([i for i in true_audio_mat_list])
true_audio_mat = true_audio_mat[:SHAPE,:]
mat = np.where(true_audio_mat < 1e-2, 1e-2, true_audio_mat)



f_mat = np.hstack([i for i in f_mat_list])
t_mat = np.hstack([i for i in t_mat_list])
r_mat = np.hstack([i for i in r_mat_list])

# f_mat = voice_fold_mat(sensor_data)
# t_mat = voice_translation_mat(sensor_data)
# r_mat = voice_recovering_mat(sensor_data)


# 预处理
mat = np.where(mat < 1e-2, 1e-2, mat)
r_mat = np.where(r_mat < 1e-2, 1e-2, r_mat)
f_mat = np.where(f_mat < 1e-2, 1e-2, f_mat)
t_mat = np.where(t_mat < 1e-2, 1e-2, t_mat)



d1 = calculate_distance(mat, f_mat)
d2 = calculate_distance(mat, t_mat)
d3 = calculate_distance(mat, r_mat)


print( "folding ", d1, "translation ", d2, "our method ", d3)
# print( "our method ", d3)
