from h import *
import pandas as pd


def csv_to_1000Hz(voice_filename,sensor_filename,save_filename):
    # read wav 降采样到1000Hz
    sampling_rate, voice = read_wav(voice_filename)
    voice = voice.copy()[::sampling_rate//1000]
    # 读取sensor数据
    df = pd.read_csv(sensor_filename, encoding='utf-8')
    x_axis = list(df["x"])
    y_axis = list(df["y"])
    z_axis = list(df["z"])

    # 500Hz升采样到1000Hz
    new_x_axis = my_interpolate(x_axis, len(x_axis) * 2)
    new_y_axis = my_interpolate(y_axis, len(y_axis) * 2)
    new_z_axis = my_interpolate(z_axis, len(z_axis) * 2)
    print("sensor old length ", len(z_axis), "sensors new length ", len(new_z_axis))

    # 过滤低于20hz的数据
    b, a = signal.butter(8, 2 * 20 / 2000, 'highpass')  # 配置滤波器 8 表示滤波器的阶数
    new_x_axis = signal.filtfilt(b, a, new_x_axis)
    new_y_axis = signal.filtfilt(b, a, new_y_axis)
    new_z_axis = signal.filtfilt(b, a, new_z_axis)

    # 保存为csv
    save_csv_df = pd.DataFrame({'x': new_x_axis, 'y': new_y_axis, 'z': new_z_axis})
    save_csv_df.to_csv(save_filename, index=False, sep=',')

    # 图形化展示
    tmp = new_z_axis.copy()[-1000*20:]
    b, a = signal.butter(8, 2 * 100 / 2000, 'highpass')  # 配置滤波器 8 表示滤波器的阶数
    tmp = signal.filtfilt(b, a, tmp)
    voice = voice[-1000*20:]
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(tmp)
    plt.subplot(2,1,2)
    plt.plot(voice)
    plt.show()
    print(save_filename)


# -----------------huawei_table_lab-----------------
import os
acc_dir = r'E:\MAG\code_final_version\data\huawei_table_lab\digit\sensor\acc'
gyr_dir = r'E:\MAG\code_final_version\data\huawei_table_lab\digit\sensor\gyr'
mag_dir = r'E:\MAG\code_final_version\data\huawei_table_lab\digit\sensor\mag'
acc_save_dir = r'E:\MAG\code_final_version\data\huawei_table_lab\digit\sensor_1000Hz\acc'
gyr_save_dir = r'E:\MAG\code_final_version\data\huawei_table_lab\digit\sensor_1000Hz\gyr'
mag_save_dir = r'E:\MAG\code_final_version\data\huawei_table_lab\digit\sensor_1000Hz\mag'
voice_dir = r'E:\MAG\code_final_version\data\huawei_table_lab\digit\voice'
sensor_data_sampling_rate = 500
for file_id in range(0,10):
    name = str(file_id)+'.csv'
    wav_name = str(file_id)
    acc_filename = os.path.join(acc_dir,name)
    gyr_filename = os.path.join(gyr_dir,name)
    mag_filename = os.path.join(mag_dir, name)
    acc_save_filename = os.path.join(acc_save_dir,name)
    gyr_save_filename = os.path.join(gyr_save_dir,name)
    mag_save_filename = os.path.join(mag_save_dir, name)
    voice_filename = os.path.join(voice_dir,wav_name+'.wav')

    # 保存对齐后的acc和gyr
    csv_to_1000Hz(voice_filename, acc_filename, acc_save_filename)
    csv_to_1000Hz(voice_filename, gyr_filename, gyr_save_filename)
    csv_to_1000Hz(voice_filename, mag_filename, mag_save_filename)