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
    z_axis = list(df["Acceleration z (m/s^2)"])

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
    intervals = list(range(-500 * 1, 500 * 11))
    intervals = list(range(3710, 3770))
    # intervals = list(range(1200, 1700))
    # 暴力寻找sensors data需要延长的长度
    for interval in tqdm(intervals):
        new_z_axis = adjust_length(z_axis, interval)
        d = calculate_time_domain_distance(new_z_axis, voice)
        if d < min_d:
            min_d = d
            min_interval = interval

    print("mind = ", min_d, " min_interval = ", min_interval)
    return min_interval


def adjust_data(min_interval, sensor_data_filename, voice_filename, save_filename, sensor_data_sampling_rate=500,is_acc=1):
    # read wav
    sampling_rate, voice = read_wav(voice_filename)

    # 读取原始sensors数据
    df = pd.read_csv(sensor_data_filename, encoding='utf-8')
    global x_axis
    global y_axis
    global z_axis
    if is_acc:
        x_axis = list(df["Acceleration x (m/s^2)"])
        y_axis = list(df["Acceleration y (m/s^2)"])
        z_axis = list(df["Acceleration z (m/s^2)"])
    else:
        x_axis = list(df["Gyroscope x (rad/s)"])
        y_axis = list(df["Gyroscope y (rad/s)"])
        z_axis = list(df["Gyroscope z (rad/s)"])

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
    sensor_len = int(time_len*500)  # sensor data的时间长度控制成与测试音频等长
    new_x_axis = new_x_axis[:sensor_len]
    new_y_axis = new_y_axis[:sensor_len]
    new_z_axis = new_z_axis[:sensor_len]
    print("voice len",time_len,"sensor len",len(new_z_axis)/500)

    # 保存为3个轴的数据为csv格式
    save_csv_df = pd.DataFrame({'x': new_x_axis, 'y': new_y_axis, 'z': new_z_axis})
    save_csv_df.to_csv(save_filename, index=False, sep=',')
    print("save csv", save_filename,len(new_z_axis),len(new_x_axis),len(new_y_axis))

    # wav 降采样到500hz
    voice = voice.copy()[::sampling_rate//500]
    voice = voice[:len(new_z_axis)]

    # 绘制传感器与voice信号图，肉眼观察数据是否对齐
    b, a = signal.butter(8, 2 * 100 / sensor_data_sampling_rate, 'highpass')  # 配置滤波器 8 表示滤波器的阶数
    tmp = signal.filtfilt(b, a, new_z_axis[300000:350000][:15011])  # data为要过滤的信号

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.title(sensor_data_filename)
    plt.plot(signal.filtfilt(b, a, tmp))
    plt.subplot(2, 1, 2)
    plt.title(voice_filename)
    plt.plot(voice[300000:350000][:15011])
    plt.show()

# -----------------robustness-----------------
# voice_filename = r'E:\MAG\code_final_version\data\robustness\voice\test_robustness.wav'
#
# name = "xiaomi"
#
# acc_filename = 'E:\\MAG\\code_final_version\\data\\robustness\\'+name+'\\raw_data\\acc\\Accelerometer.csv'
# gyr_filename = 'E:\\MAG\\code_final_version\\data\\robustness\\'+name+'\\raw_data\\gyr\\Gyroscope.csv'
#
# acc_save_filename = 'E:\\MAG\\code_final_version\\data\\robustness\\'+name+'\\aligned\\acc\\Accelerometer.csv'
# gyr_save_filename = 'E:\\MAG\\code_final_version\\data\\robustness\\'+name+'\\aligned\\gyr\\Gyroscope.csv'
#
# sensor_data_sampling_rate = 397
# # 对齐acc和voice
# min_interval = auto_align(acc_filename, voice_filename, sensor_data_sampling_rate)
# # 保存对齐后的acc和gyr
# adjust_data(min_interval, acc_filename, voice_filename, acc_save_filename, sensor_data_sampling_rate)
# adjust_data(min_interval, gyr_filename, voice_filename, gyr_save_filename, sensor_data_sampling_rate, is_acc=0)

# # -----------------test_subject-----------------
# import os
# acc_dir = r'E:\MAG\code_final_version\data\test_subject\raw_data\acc'
# gyr_dir = r'E:\MAG\code_final_version\data\test_subject\raw_data\gyr'
# acc_save_dir = r'E:\MAG\code_final_version\data\test_subject\aligned\acc'
# gyr_save_dir = r'E:\MAG\code_final_version\data\test_subject\aligned\gyr'
# voice_dir = r'E:\MAG\code_final_version\data\test_subject\voice'
# sensor_data_sampling_rate = 500
# for file_id in range(11,21):
#     name = str(file_id)+'.csv'
#     acc_filename = os.path.join(acc_dir,name)
#     gyr_filename = os.path.join(gyr_dir,name)
#     acc_save_filename = os.path.join(acc_save_dir,name)
#     gyr_save_filename = os.path.join(gyr_save_dir,name)
#     voice_filename = os.path.join(voice_dir,str(file_id)+'.wav')
#     # 对齐acc和voice
#     min_interval = auto_align(acc_filename, voice_filename, sensor_data_sampling_rate)
#     # 保存对齐后的acc和gyr
#     adjust_data(min_interval, acc_filename, voice_filename, acc_save_filename, sensor_data_sampling_rate)
#     adjust_data(min_interval, gyr_filename, voice_filename, gyr_save_filename, sensor_data_sampling_rate, is_acc=0)

# # -----------------in_the_wild-----------------
# import os
# acc_dir = r'E:\MAG\code_final_version\data\in_the_wild\raw_data\acc'
# gyr_dir = r'E:\MAG\code_final_version\data\in_the_wild\raw_data\gyr'
# acc_save_dir = r'E:\MAG\code_final_version\data\in_the_wild\aligned\acc'
# gyr_save_dir = r'E:\MAG\code_final_version\data\in_the_wild\aligned\gyr'
# voice_dir = r'E:\MAG\code_final_version\data\in_the_wild\voice'
# sensor_data_sampling_rate = 500
# names = ['DR1','DR3']
# for file_id in names:
#     name = str(file_id)+'.csv'
#     acc_filename = os.path.join(acc_dir,name)
#     gyr_filename = os.path.join(gyr_dir,name)
#     acc_save_filename = os.path.join(acc_save_dir,name)
#     gyr_save_filename = os.path.join(gyr_save_dir,name)
#     voice_filename = os.path.join(voice_dir,str(file_id)+'.wav')
#     # 对齐acc和voice
#     min_interval = auto_align(acc_filename, voice_filename, sensor_data_sampling_rate)
#     # 保存对齐后的acc和gyr
#     adjust_data(min_interval, acc_filename, voice_filename, acc_save_filename, sensor_data_sampling_rate)
#     adjust_data(min_interval, gyr_filename, voice_filename, gyr_save_filename, sensor_data_sampling_rate, is_acc=0)

# -----------------huawei_table_lab-----------------
# import os
# acc_dir = r'E:\MAG\code_final_version\data\huawei_table_lab\raw_data\acc'
# gyr_dir = r'E:\MAG\code_final_version\data\huawei_table_lab\raw_data\gyr'
# acc_save_dir = r'E:\MAG\code_final_version\data\huawei_table_lab\aligned\acc'
# gyr_save_dir = r'E:\MAG\code_final_version\data\huawei_table_lab\aligned\gyr'
# voice_dir = r'E:\MAG\code_final_version\data\huawei_table_lab\voice'
# sensor_data_sampling_rate = 500
# for file_id in range(1,11):
#     name = str(file_id)+'.csv'
#     wav_name = "%02d" % file_id
#     acc_filename = os.path.join(acc_dir,name)
#     gyr_filename = os.path.join(gyr_dir,name)
#     acc_save_filename = os.path.join(acc_save_dir,name)
#     gyr_save_filename = os.path.join(gyr_save_dir,name)
#     voice_filename = os.path.join(voice_dir,wav_name+'.wav')
#     # 对齐acc和voice
#     min_interval = auto_align(acc_filename, voice_filename, sensor_data_sampling_rate)
#     # 保存对齐后的acc和gyr
#     adjust_data(min_interval, acc_filename, voice_filename, acc_save_filename, sensor_data_sampling_rate)
#     adjust_data(min_interval, gyr_filename, voice_filename, gyr_save_filename, sensor_data_sampling_rate, is_acc=0)

# -----------------robustness more devices-----------------
voice_filename = r'E:\MAG\code_final_version\data\robustness\voice\test_robustness.wav'

# name = "huawei_p20_pro"
# name = "huawei_honor_v30"
name = "motorala_edge_S"
# name = "redmi_k30_pro"
# name = "VIVO_IQOO3"
# name = "samsung_S20_Ultra"
# name = "samsung_S21"

# name = "oppo"
# name = "xiaomi"

acc_filename = 'E:\\MAG\\code_final_version\\data\\robustness\\'+name+'\\raw_data\\acc\\Accelerometer.csv'
gyr_filename = 'E:\\MAG\\code_final_version\\data\\robustness\\'+name+'\\raw_data\\gyr\\Gyroscope.csv'

acc_save_filename = 'E:\\MAG\\code_final_version\\data\\robustness\\'+name+'\\aligned\\acc\\Accelerometer.csv'
gyr_save_filename = 'E:\\MAG\\code_final_version\\data\\robustness\\'+name+'\\aligned\\gyr\\Gyroscope.csv'

# sensor_data_sampling_rate = 500
# sensor_data_sampling_rate = 500
sensor_data_sampling_rate = 497
# sensor_data_sampling_rate = 405
# sensor_data_sampling_rate = 425
# sensor_data_sampling_rate = 392
# sensor_data_sampling_rate = 405

# sensor_data_sampling_rate = 418
# sensor_data_sampling_rate = 397

# 对齐acc和voice
min_interval = auto_align(acc_filename, voice_filename, sensor_data_sampling_rate)
# 保存对齐后的acc和gyr
adjust_data(min_interval, acc_filename, voice_filename, acc_save_filename, sensor_data_sampling_rate)
adjust_data(min_interval, gyr_filename, voice_filename, gyr_save_filename, sensor_data_sampling_rate, is_acc=0)

