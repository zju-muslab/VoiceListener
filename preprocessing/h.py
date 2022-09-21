import serial
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile
import os


NFFT_S = 20
NOVERLAT_S = 40
SHAPE = 51
low, high = 4, 10
SR = 500
NEW_SR = 2000

NFFT_S = 10
NOVERLAT_S = 15
SHAPE = 257
low, high = 30, 76
SR = 500
NEW_SR = 2000




# NFFT_S = 10
# NOVERLAT_S = 20
# SHAPE = 101
# low, high = 80, 160
# SR = 500
# NEW_SR = 2000
# BAND = 80   # mag 20 acc 80
# MIN_SUM = 0.002 # mag 15 acc 0.0020
# SENSORNAME = 'mag'
# plt.rcParams["font.family"] = "Times New Roman"


def my_interpolate(my_signal,length):
    np.linspace(0,len(my_signal),len(my_signal))
    # np.interp(,,a)
    x = np.linspace(0,len(my_signal),len(my_signal),endpoint=False)
    new_x = np.linspace(0,len(my_signal),length,endpoint=False)
    ans = np.interp(new_x,x,my_signal)  # 新的x轴  原信号x轴 原信号y轴
    return ans

def read_wav(filename):
    sampling_freq, audio = wavfile.read(filename)
    return sampling_freq, audio

def save_json(filename,data):
    save_json = filename
    with open(save_json, 'w') as f:
        json.dump(data, f)


def get_base_freq_modified(mag_magnitude,low,high, MIN_SUM):
    m, n = mag_magnitude.shape
    p = np.zeros((m, n))  # 记录每个时间点内左右可能是谐波的点
    for j in range(n):
        for i in range(low,high):
            now_i = i  # now_i 为目前判定为基频的频率值
            # now_sum = get_nearby_m(mag_magnitude, now_i, j)
            idx = 2
            hs = np.array(list(range(1,idx)))*now_i
            new_hs = []
            for hs_i in hs:
                if hs_i <= mag_magnitude.shape[0]:
                    # new_hs.append(hs_i)
                    for idx in range(-2,3):
                        hs_idx = hs_i+idx
                        if 0 < hs_idx < mag_magnitude.shape[0]:
                            new_hs.append(hs_idx)
                    # if hs_i-1>0:
                    #     new_hs.append(hs_i-1)
                    # if hs_i+1<mag_magnitude.shape[0]:
                    #     new_hs.append(hs_i+1)
            hs = new_hs
            hs = np.array((hs)).reshape(-1)
            # hs = [get_new_freq(int((mag_magnitude.shape[0]-1)/4), i) for i in hs]
            now_sum = 0
            for id_num,h in enumerate(hs):
                now_sum += mag_magnitude[h][j]/(id_num+1)
            if now_sum < MIN_SUM:  # acc 0.005 mag5
                now_sum = 0
            p[i][j] = now_sum


    harmonics = np.argmax(p, axis=0)
    harmonics = np.array(harmonics, dtype='int')
    #
    x = np.array([i for i in range(-10,10)])
    new_harmonics = np.zeros(harmonics.shape)
    for idx, i in enumerate(new_harmonics):
        if harmonics[idx] == 0:
            continue
        new_idxs = x+idx
        tmp_list = []
        for new_idx in new_idxs:
            if 0 < new_idx < len(new_harmonics):
                if harmonics[new_idx]!=0:
                    # print(new_idx,harmonics[new_idx])
                    tmp_list.append(harmonics[new_idx])
        # print(tmp_list)
        if len(tmp_list)>0:
            new_harmonics[idx] = int(np.mean(tmp_list))


    # plt.figure()
    # plt.plot(new_harmonics)
    # plt.show()

    # for idx,i in enumerate(harmonics):
    #     if i!=0:
    #         if harmonics[]
    #         begin_idx = idx - 2
    #         end_idx = idx + 2
    #         if begin_idx < 0:
    #             begin_idx = 0
    #         if end_idx>len(harmonics):
    #             end_idx = len(harmonics)
    #         tmp_harmonics = harmonics[begin_idx:end_idx]
    #         harmonics[idx] = int(np.mean(harmonics))

    # minus_harmonics = []
    # for idx,i in enumerate(harmonics):


    # plt.figure()
    # plt.plot(harmonics)
    # plt.show()


    return new_harmonics
    # return harmonics


def get_magnitude_mat(Zxx):
    magnitude_mat = np.abs(Zxx)
    return magnitude_mat

def read_json(filename):
    with open(filename, 'r', encoding='utf8')as fp:
        json_data = json.load(fp)
        return json_data

def to_12bit_int(raw_data):
    """
    接收到16bit数据，去除4bit最低位没用到的数据
    把12bit的16进制字符串转化成12bit的补码数值

    :param raw_data: 16进制字符串数据
    :return:  磁感应强度 单位mG
    """
    # print(raw_data)
    ans = 0
    if len(raw_data) == 4:
        raw_data = raw_data[0:-1]
    raw_data = bin(int(raw_data, 16))
    raw_data = raw_data[2:]

    if len(raw_data) != 12:
        for i in range(12-len(raw_data)):
            raw_data = "0" + raw_data
    for i in str(raw_data)[1:]:
        ans = ans * 2 + int(i)
    if int(str(raw_data)[0]) == 0:
        ans -= 2048
    return ans*7.8125/10

def add_zero(s):
    new_s = []
    for i in s:
        new_s.append(i)
        new_s.append(0)
    return new_s

def mul_minus1(s):
    new_s = []
    for idx,i in enumerate(s):
        new_s.append(i*pow(-1,idx+1))
    return new_s

def spectrum_transaltion(s):
    new_s0 = add_zero(s)
    new_s1 = mul_minus1(s)
    new_s1 = add_zero(new_s1)
    b, a = signal.butter(8, 0.5, 'lowpass')  # 配置滤波器 8 表示滤波器的阶数
    new_s0 = signal.filtfilt(b, a, new_s0)  # data为要过滤的信号
    b, a = signal.butter(8, 0.5, 'highpass')  # 配置滤波器 8 表示滤波器的阶数
    new_s1 = signal.filtfilt(b, a, new_s1)  # data为要过滤的信号
    res = new_s0+new_s1
    # f, t, Zxx = signal.stft(res, fs=sr*2, nperseg=256, noverlap=128)
    # mat = get_magnitude_mat(Zxx)
    # my_plot(mat,sr)
    return res

def my_plot(mat, y_len):
    plt.figure()
    plt.pcolormesh(np.linspace(0, mat.shape[1], mat.shape[1]),np.linspace(0, y_len, mat.shape[0]), mat, shading='auto', cmap="magma")
    # plt.colorbar()
    plt.show()


def get_new_freq(m, old_freq):
    if old_freq == m*2:
        return 0
    if old_freq == m:
        return m
    if old_freq % m == 0:
        if old_freq/m%2 == 0:
            return 0
        else:
            return m
    while old_freq > m:
        num = int(old_freq/m)
        num = num*m*2
        old_freq = num - old_freq
        # print(old_freq)
    return old_freq


def fold_mat(mat):
    """
    fold the mat
    :param mat:
    :return:
    """
    # print("mat",mat)
    tmp_mat = mat[::-1]
    test_mat = np.vstack((mat[:-1, :], mat[-1, :], tmp_mat[1:, :]))
    test_mat = np.vstack((test_mat[:-1, :], test_mat[-1, :], test_mat[::-1][1:, :]))
    # test_mat = np.vstack((test_mat[:-1, :], test_mat[-1, :], test_mat[::-1][1:, :]))
    # test_mat = np.vstack((test_mat[:-1, :], test_mat[-1, :], test_mat[::-1][1:, :]))
    test_mat = test_mat[:SHAPE,:]
    return test_mat


def read_mag(mag_data):
    mag = []
    for i in mag_data.split(" ")[:-1]:  # 去掉最后一个回车
        if len(i) > 0:
            # print(to_12bit_int(i))
            mag.append(to_12bit_int(i))
    # plt.figure()
    # plt.plot(mag)
    # plt.show()
    save_json("d:/test.json",{"data":mag})
    b, a = signal.butter(8, 2 * 20 / 500, 'highpass')  # 配置滤波器 8 表示滤波器的阶数
    mag = signal.filtfilt(b, a, mag)  # data为要过滤的信号
    f, t, Zxx = signal.stft(mag, fs=SR, nperseg=int(SR // NFFT_S), noverlap=int(SR // NOVERLAT_S))
    return get_magnitude_mat(Zxx)


def GLA(S, n_iter=100, n_fft=4096, hop_length=None, window='hann',fs=1000):
    hop_length = n_fft//4 if hop_length is None else hop_length
    m_phase = np.exp(2j*np.pi*np.random.rand(*S.shape))
    for i in range(n_iter):
        xi = np.abs(S).astype(np.complex)*m_phase  # 原始幅度谱与相位谱组合成ZXX
        m_signal = signal.istft(xi, fs, nfft=n_fft*2,nperseg=n_fft,noverlap=hop_length,boundary = None)[1]   # 使用Zxx还原回时域信号

        next_xi = signal.stft(m_signal, fs, nfft=n_fft*2,nperseg=n_fft,noverlap=hop_length,padded = False, boundary = None)[2]

        m_phase = np.exp(1j*np.angle(next_xi))  # 取相位

    xi = np.abs(S).astype(np.complex)*m_phase
    m_signal = signal.istft(xi, fs, nperseg=n_fft,noverlap=hop_length)[1]
    return m_signal


def save_wav(filename,sampling_freq,my_signal):
    my_signal = np.array(my_signal,dtype="float")
    my_signal /= my_signal.max()
    # plt.figure()
    # plt.plot(my_signal)
    # plt.show()
    my_signal *= np.iinfo(np.int32).max
    my_signal = np.asarray(my_signal, dtype=np.int32)
    wavfile.write(filename, sampling_freq, my_signal)

def sum_xy(xs, ys, mat):
    mat_shape = mat.shape
    tmp_list = []
    for x, y in zip(xs,ys):
        if 0 <x<mat_shape[0] and 0<y<mat_shape[1]:
            tmp_list.append(mat[x][y])
    return np.mean(tmp_list)

# 梅尔频谱
f = np.array(range(1,1001))
mel_f = 2595*np.log(1+f/700)
mel_f = np.max(mel_f)-mel_f
mel_f /=np.max(mel_f)

def get_new_mat(mat, p):
    m, n = mat.shape
    new_mat = np.zeros((SHAPE, n))
    # 原版恢复幅度谱
    # plt.figure()
    # plt.pcolormesh(np.linspace(0, mat.shape[1], mat.shape[1]),np.linspace(0, 250, mat.shape[0]), mat, shading='auto', cmap="magma")
    # plt.colorbar()
    # plt.show()
    x_plus = np.array([i for i in range(-3,4)],dtype=np.int)
    y_plus = np.array([i for i in range(-3,4)],dtype=np.int)



    for j, i in enumerate(p):
        if i != 0:
            for idx in range(1, 60):
                for x in range(-5, 6):
                    new_freq = get_new_freq(m, (i) * idx + x)
                    if new_freq < m and 0 <= int(i * idx + x) < new_mat.shape[0] - 1:
                        new_i = int((i) * idx + x)
                        new_mat[new_i][j] = mat[new_i][j]
                        # new_mat[new_i][j] = sum_xy(new_i+x_plus,j+y_plus,mat)
                        # if idx != 1: # 如果是第一条谐波
                        #     if new_mat[new_i][j] > new_mat[int((i) * (idx-1) + x)][j]:
                        #         # new_mat[new_i][j]/=4
                        #         # print("!!!!")
                        #         pass







    # 增加 谐波能量平滑功能


    # ans_mat = np.zeros((SHAPE, n))
    # for i in range(ans_mat.shape[0]):
    #     for j in range(ans_mat.shape[1]):
    #        if mat[i][j]>1e-8:
    #            new_is = i + x_plus
    #            new_js = j + y_plus
    #            tmp_list = []
    #            for new_i,new_j in zip(new_is,new_js):
    #                if 0< new_i <ans_mat.shape[0] and 0< new_j<ans_mat.shape[1]:
    #                    if mat[new_i][new_j]>1e-8:
    #                         tmp_list.append(mat[new_i][new_j])
    #
    #            # print(tmp_list)
    #            ans_mat[i][j] = np.sum(tmp_list)
    #            # ans_mat[i][j] = new_mat[i][j]

    # plt.figure()
    # plt.subplot(1, 3, 1)
    # plt.pcolormesh(np.linspace(0, mat.shape[1], mat.shape[1]),np.linspace(0, 250, mat.shape[0]), mat, shading='auto', cmap="magma")
    # plt.colorbar()
    #
    # plt.subplot(1, 3, 2)
    # plt.pcolormesh(np.linspace(0, mat.shape[1], mat.shape[1]), np.linspace(0, 1000, new_mat.shape[0]), new_mat, shading='auto', cmap="magma")
    # plt.colorbar()
    # return ans_mat
    return new_mat


def read_serial(serialPort="COM4",baudRate=115200):
    flg = False
    ser = serial.Serial(serialPort, baudRate, timeout=0.5)
    print("参数设置：串口=%s ，波特率=%d" % (serialPort, baudRate))
    while 1:
        s = ser.readline()
        # if len(s) != 0 and s[0] != 'x':
        s = str(s, "utf-8")
        if len(s) > 10 and s[0] == 'M' and s[1] == 'a' and s[2] == 'g':
            print("磁力计读取数据中")
            flg = True
        if len(s) > 10000:
            # print(s)
            flg = False
            print("\n正在恢复语音数据")
            mat = read_mag(s)
            # my_plot(mat, 250)
            p = get_base_freq_modified(fold_mat(mat), low, high)
            new_mat = get_new_mat(mat, p)
            # my_plot(new_mat, 1000)
            origin_signal = GLA(new_mat, n_fft=int(NEW_SR // NFFT_S), n_iter=100, hop_length=int(NEW_SR / NOVERLAT_S),
                                fs=NEW_SR)
            save_wav("d:/test.wav", int(NEW_SR), origin_signal)
            # song = AudioSegment.from_wav("d:/test.wav")
            # play(song)
            os.system("d:/test.wav")

        if flg and len(s) == 0:
            print("-> ", end="")

# os.system("d:/test.wav")
# read_serial()



