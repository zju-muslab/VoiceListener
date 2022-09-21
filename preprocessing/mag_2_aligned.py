from h import *
from tqdm import tqdm
import pandas as pd


def adjust_length(s,num):
    if num < 0:
        return s[-num:]
    else:
        return [s[0] for i in range(num)] + s


def calculate_time_domain_distance(sensor_data, voice):
    sensor_data = np.array(sensor_data,dtype=np.float)
    b, a = signal.butter(8, 2 * 100 / 500, 'highpass')  # 配置滤波器 8 表示滤波器的阶数
    sensor_data = signal.filtfilt(b, a, sensor_data)  # data为要过滤的信号

    voice = np.array(voice,dtype=np.float)
    # 标准化
    sensor_data -= np.min(sensor_data)
    sensor_data /= np.max(sensor_data)
    voice -= np.min(voice)
    voice /= np.max(voice)
    voice = voice[:500*80]
    sensor_data = sensor_data[:500*80]
    return np.sum(np.abs(voice-sensor_data))


def auto_align(sensor_data_filename, voice_filename, sensor_data_sampling_rate=500):

    print("read csv", sensor_data_filename)

    # read csv and record acc data
    df = pd.read_csv(sensor_data_filename, encoding='utf-8')
    z_axis = list(df["z"])

    # 如果sensor的采样率低于500Hz, 将其线性插值到500Hz
    if sensor_data_sampling_rate != 500:
        z_axis = list(my_interpolate(z_axis, int(len(z_axis) * 1.0 /sensor_data_sampling_rate * 500)))

    # read wav
    sampling_rate, voice = read_wav(voice_filename)
    # wav 降采样到500hz
    voice = voice.copy()[::sampling_rate//500]
    voice = voice[:len(z_axis)]

    # 对齐sensor和voice
    min_d = 1e8
    global min_interval
    min_interval = 0
    # 设置移动传感器数据的长度
    intervals = list(range(-500 * 1, 500 * 5))

    # 暴力寻找sensors data需要延长的长度
    for interval in tqdm(intervals):
        new_z_axis = adjust_length(z_axis, interval)
        d = calculate_time_domain_distance(new_z_axis, voice)
        if d < min_d:
            min_d = d
            min_interval = interval

    print("mind = ", min_d, " min_interval = ", min_interval)
    return min_interval


def adjust_data(min_interval, sensor_data_filename, voice_filename, save_filename, sensor_data_sampling_rate=500):
    # read wav
    sampling_rate, voice = read_wav(voice_filename)

    # 读取原始sensors数据
    df = pd.read_csv(sensor_data_filename, encoding='utf-8')


    x_axis = list(df["x"])
    y_axis = list(df["y"])
    z_axis = list(df["z"])

    # 如果sensor的采样率低于500Hz, 将其线性插值到500Hz
    if sensor_data_sampling_rate != 500:
        print(sensor_data_filename," to 500Hz ","origin_len",len(x_axis), end=" ")
        x_axis = list(my_interpolate(x_axis, int(len(x_axis) * 1.0 /sensor_data_sampling_rate * 500)))
        print("new_len", len(x_axis))
        y_axis = list(my_interpolate(y_axis, int(len(y_axis) * 1.0 /sensor_data_sampling_rate * 500)))
        z_axis = list(my_interpolate(z_axis, int(len(z_axis) * 1.0 /sensor_data_sampling_rate * 500)))

    # 调整3个轴的数据
    new_x_axis = adjust_length(x_axis, min_interval)
    new_y_axis = adjust_length(y_axis, min_interval)
    new_z_axis = adjust_length(z_axis, min_interval)

    # 尾部与voice对齐
    time_len = len(voice)/sampling_rate
    sensor_len = int(time_len*sensor_data_sampling_rate)
    new_x_axis = new_x_axis[:sensor_len]
    new_y_axis = new_y_axis[:sensor_len]
    new_z_axis = new_z_axis[:sensor_len]
    print("voice len",time_len,"sensor len",len(new_z_axis)/sensor_data_sampling_rate)

    # 保存为3个轴的数据为csv格式
    save_csv_df = pd.DataFrame({'x': new_x_axis, 'y': new_y_axis, 'z': new_z_axis})
    save_csv_df.to_csv(save_filename, index=False, sep=',')
    print("save csv", save_filename)

    # wav 降采样到500hz
    voice = voice.copy()[::sampling_rate//500]
    voice = voice[:len(new_z_axis)]

    # 绘制传感器与voice信号图，肉眼观察数据是否对齐
    b, a = signal.butter(8, 2 * 100 / sensor_data_sampling_rate, 'highpass')  # 配置滤波器 8 表示滤波器的阶数
    tmp = signal.filtfilt(b, a, new_z_axis[:int(sensor_data_sampling_rate * 100)])  # data为要过滤的信号

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.title(sensor_data_filename)
    plt.plot(signal.filtfilt(b, a, tmp))
    plt.subplot(2, 1, 2)
    plt.title(voice_filename)
    plt.plot(voice[:int(sensor_data_sampling_rate * 100)])
    plt.show()

# # -----------------robustness-----------------
# voice_filename = r'E:\MAG\code_final_version\data\robustness\voice\test_robustness.wav'
#
# name = 'xiaomi'
#
# mag_filename = 'E:\\MAG\\code_final_version\\data\\robustness\\'+name+'\\raw_data\\mag_csv\\mag.csv'
# mag_save_filename = 'E:\\MAG\\code_final_version\\data\\robustness\\'+name+'\\aligned\\mag\\mag.csv'
# sensor_data_sampling_rate = 500
#
# # 对齐mag
# min_interval = auto_align(mag_filename, voice_filename, sensor_data_sampling_rate)
# # 保存对齐后的mag
# adjust_data(min_interval, mag_filename, voice_filename, mag_save_filename, sensor_data_sampling_rate)


# # -----------------test_subject-----------------
# import os
# mag_dir = r'E:\MAG\code_final_version\data\test_subject\raw_data\mag_csv'
# mag_save_dir = r'E:\MAG\code_final_version\data\test_subject\aligned\mag'
# voice_dir = r'E:\MAG\code_final_version\data\test_subject\voice'
#
# sensor_data_sampling_rate = 500
# for file_id in range(11,21):
#     name = str(file_id)+'.csv'
#     mag_filename = os.path.join(mag_dir,name)
#     mag_save_filename = os.path.join(mag_save_dir,name)
#     voice_filename = os.path.join(voice_dir,str(file_id)+'.wav')
#     # 对齐mag
#     min_interval = auto_align(mag_filename, voice_filename, sensor_data_sampling_rate)
#     # 保存对齐后的mag
#     adjust_data(min_interval, mag_filename, voice_filename, mag_save_filename, sensor_data_sampling_rate)


# # -----------------in_the_wild-----------------
# import os
# mag_dir = r'E:\MAG\code_final_version\data\in_the_wild\raw_data\mag_csv'
# mag_save_dir = r'E:\MAG\code_final_version\data\in_the_wild\aligned\mag'
# voice_dir = r'E:\MAG\code_final_version\data\in_the_wild\voice'
# names = ['DR1', 'DR3']
# sensor_data_sampling_rate = 500
#
# for file_id in names:
#     name = str(file_id)+'.csv'
#     mag_filename = os.path.join(mag_dir,name)
#     mag_save_filename = os.path.join(mag_save_dir,str(file_id)+'.csv')
#     voice_filename = os.path.join(voice_dir,str(file_id)+'.wav')
#     # 对齐mag
#     min_interval = auto_align(mag_filename, voice_filename, sensor_data_sampling_rate)
#     # 保存对齐后的mag
#     adjust_data(min_interval, mag_filename, voice_filename, mag_save_filename, sensor_data_sampling_rate)

# -----------------huawei_table_lab-----------------
import os
mag_dir = r'E:\MAG\code_final_version\data\huawei_table_lab\raw_data\mag_csv'
mag_save_dir = r'E:\MAG\code_final_version\data\huawei_table_lab\aligned\mag'
voice_dir = r'E:\MAG\code_final_version\data\huawei_table_lab\voice'

sensor_data_sampling_rate = 500
for file_id in range(1,11):
    name = str(file_id)+'.csv'
    wav_name = "%02d" % file_id
    mag_filename = os.path.join(mag_dir,name)
    mag_save_filename = os.path.join(mag_save_dir,str(file_id)+'.csv')
    voice_filename = os.path.join(voice_dir,wav_name+'.wav')

    # 对齐mag
    min_interval = auto_align(mag_filename, voice_filename, sensor_data_sampling_rate)
    # 保存对齐后的mag
    adjust_data(min_interval, mag_filename, voice_filename, mag_save_filename, sensor_data_sampling_rate)