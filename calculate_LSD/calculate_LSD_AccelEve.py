from h import *
import pandas as pd
import scipy.io as sio
import os
import matplotlib


def read_matlab_mat(filename):
    ans = sio.loadmat(filename)
    return ans["data"]

def get_log_power_mat(magnitude_mat):
    return np.log(magnitude_mat*magnitude_mat)


def calculate_distance(mat,recover_mat):
    # points = []
    # mat_sum = np.sum(mat, axis=0)
    # for idx, i in enumerate(mat_sum):
    #     if i > 3:
    #         points.append(idx)
    # mat = mat[:,points]
    # recover_mat = recover_mat[:, points]

    mat = get_log_power_mat(mat)

    recover_mat = get_log_power_mat(recover_mat)

    d_mat = mat-recover_mat
    k, l = d_mat.shape
    d_mat = d_mat*d_mat
    d_sum = np.sum(d_mat, axis=0)
    d_sum = np.sqrt(d_sum/k)
    result = np.sum(d_sum)/l

    mat = np.flip(mat,1)
    recover_mat = np.flip(recover_mat,1)

    # mat = mat[:,:11050]
    # recover_mat = recover_mat[:, :11050]

    mat = mat[:, :recover_mat.shape[1]]

    plt.figure()
    plt.subplot(2,1,1)
    plt.pcolormesh(np.linspace(0, mat.shape[1], mat.shape[1]), np.linspace(0, mat.shape[0], mat.shape[0]),mat,shading='auto', cmap="magma")
    plt.colorbar()
    plt.subplot(2,1,2)
    plt.colorbar()
    plt.pcolormesh(np.linspace(0, recover_mat .shape[1],recover_mat .shape[1]), np.linspace(0, recover_mat.shape[0], recover_mat .shape[0]),recover_mat ,shading='auto', cmap="magma")
    plt.show()
    return result

# mat_dir = r'E:\AccelEve-wl-version\network_results\Acc20\mat'
audio_mat_dir = r'E:\AccelEve-wl-version\network_results\Acc20_lab_10\audio_mat'
mat_dir = r'E:\AccelEve-wl-version\network_results\Acc20_lab_10\mat'
# audio_mat_dir = r'E:\AccelEve-wl-version\network_results\Acc20_less\audio_mat'
# audio_mat_dir = r'E:\AccelEve-wl-version\network_results\robustness\audio\mat'
# mat_dir = r'E:\AccelEve-wl-version\network_results\robustness\huawei_handhold\mat'
# audio_filename = r'E:\AccelEve-wl-version\Experimental_Data\robustness_raw_data\test_robustness.wav'
sum = 0.0
mat_list = []
audio_mat_list = []
for i in range(462):
    mat_filename = os.path.join(mat_dir,str(i)+".mat")
    # print(mat_filename)
    audio_mat_filename = os.path.join(audio_mat_dir, str(i)+'.mat')

    mat = read_matlab_mat(mat_filename)
    audio_mat = read_matlab_mat(audio_mat_filename)
    # 归一化
    if np.max(mat)>1e-5:
        mat /= np.max(mat)
    if np.max(audio_mat)>1e-5:
        audio_mat /= np.max(audio_mat)

    mat = np.where(mat < 1e-2, 1e-2, mat)
    audio_mat = np.where(audio_mat < 1e-2, 1e-2, audio_mat)

    # mat = mat[::-1]
    # audio_mat = audio_mat[::-1]
    # mat = mat[:256, :]
    # audio_mat = audio_mat[:256, :]
    # print(calculate_distance(audio_mat,mat))

    # mat = mat[::-1]
    # audio_mat = audio_mat[::-1]
    # mat = mat[:257, :]
    # audio_mat = audio_mat[:257, :]
    # tmp_signal = GLA(mat, n_iter=100, n_fft=128 * 2, hop_length=120 * 2, window='hann', fs=100)
    # save_wav("d:/accel_testnow.wav", 2000, tmp_signal)
    # plt.figure()
    # plt.pcolormesh(np.linspace(0, mat.shape[1], mat.shape[1]), np.linspace(0, mat.shape[0], mat.shape[0]),np.log(mat),shading='auto', cmap="magma")
    # plt.colorbar()
    # plt.show()

    # 计算LSD
    # 去掉空白
    print(np.sum(audio_mat))
    # 去掉空白的语音
    # if np.sum(audio_mat)<500:
    #     continue

    mat_list.append(mat)
    audio_mat_list.append(audio_mat)


mat = np.hstack((i for i in mat_list))
audio_mat = np.hstack((i for i in audio_mat_list))

# 只保留1000hz以下

mat = mat[::-1]
audio_mat = audio_mat[::-1]

mat = mat[:256,:]
audio_mat = audio_mat[:256,:]

# mat = np.zeros(mat.shape)+1e-2


print(mat.shape)

# print("new_mat shape",mat.shape,"new_audio_mat shape", audio_mat.shape)
# python读取的mat
# read_audio
# sampling_rate, audio = read_wav(audio_filename)
# true_audio_mat_list = []
# for i in range(570):
#     begin = int(i*55200)
#     end = int((i+1)*55200)
#     print(begin,end)
#     tmp_audio = audio.copy()[begin:end]
#     print(len(tmp_audio))
#     f, t, Zxx = signal.stft(tmp_audio, fs=sampling_rate, nfft=128*48*2,nperseg=128*48, noverlap=120*48, padded=False,boundary=None) #对应AccelEve的audio2img中的specgram函数
#     tmp_mat = get_magnitude_mat(Zxx)
#     print(tmp_mat.shape)
#     tmp_mat = tmp_mat[:256:]
#
#
#     # s,freqs, t = matplotlib.mlab.specgram(tmp_audio, Fs=sampling_rate, NFFT=128 * 48,
#     #                                      noverlap=120 * 48)
#     # tmp_mat = s[:256,:]
#
#     print("shape",tmp_mat.shape)
#     if np.max(tmp_mat)>1e-5:
#         tmp_mat /= np.max(tmp_mat)
#
#     # tmp_mat = 255*(tmp_mat-np.min(tmp_mat))*(np.max(tmp_mat)-np.min(tmp_mat))
#     # tmp_mat = np.uint8(tmp_mat)
#     true_audio_mat_list.append(tmp_mat)
#
# true_audio_mat = np.hstack([i for i in true_audio_mat_list])
# true_audio_mat = true_audio_mat[:256,:]
# true_audio_mat = np.where(true_audio_mat < 1e-2, 1e-2, true_audio_mat)
# 计算LSD
# sum2 = calculate_distance(true_audio_mat,mat)

mat = np.zeros(mat.shape)+1e-2

sum = calculate_distance(audio_mat,mat)

mat = mat[:,128*50:128*57]
audio_mat = audio_mat[:,128*50:128*57]
# true_audio_mat = true_audio_mat[:,128*50:128*57]
# print("tmp",tmp_sum,"tot",sum)
plt.figure()
plt.subplot(2,1,1)
plt.title("AccelEve")
plt.pcolormesh(np.linspace(0, mat.shape[1], mat.shape[1]), np.linspace(0, mat.shape[0], mat.shape[0]),np.log(mat), cmap="magma")
plt.colorbar()


plt.subplot(2,1,2)
plt.title("ground truth")
plt.colorbar()
plt.pcolormesh(np.linspace(0, audio_mat.shape[1], audio_mat.shape[1]), np.linspace(0, audio_mat.shape[0], audio_mat.shape[0]),np.log(audio_mat), cmap="magma")

# plt.subplot(3,1,3)
# plt.colorbar()
# plt.pcolormesh(np.linspace(0, true_audio_mat.shape[1], true_audio_mat.shape[1]), np.linspace(0, true_audio_mat.shape[0], true_audio_mat.shape[0]),np.log(true_audio_mat), cmap="magma")

plt.show()

print(sum,sum/1586)
