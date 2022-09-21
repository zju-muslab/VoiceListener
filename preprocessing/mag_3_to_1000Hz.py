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
    tmp = new_z_axis.copy()#[:1000*150]
    min_len = int(np.min((len(tmp),voice.shape[0])))
    tmp = tmp[:min_len]
    voice = voice.copy()[:min_len]

    # tmp = tmp[540000:580000]
    # voice = voice.copy()[540000:580000]
    b, a = signal.butter(8, 2 * 100 / 2000, 'highpass')  # 配置滤波器 8 表示滤波器的阶数
    tmp = signal.filtfilt(b, a, tmp)

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(tmp)
    plt.subplot(2,1,2)
    plt.plot(voice)
    plt.show()
    print(save_filename)

# # -----------------robustness-----------------
# name = 'huawei_tap'
#
# voice_filename = r'E:\MAG\code_final_version\data\robustness\voice\test_robustness.wav'
#
# mag_filename = 'E:\\MAG\\code_final_version\\data\\robustness\\'+name+'\\aligned\\mag\\mag.csv'
# mag_save_filename = 'E:\\MAG\\code_final_version\\data\\robustness\\'+name+'\\aligned_1000Hz\\mag\\mag.csv'
#
# csv_to_1000Hz(voice_filename, mag_filename, mag_save_filename)


# # -----------------test_subject-----------------
# import os
# mag_dir = r'E:\MAG\code_final_version\data\test_subject\aligned\mag'
# mag_save_dir = r'E:\MAG\code_final_version\data\test_subject\aligned_1000Hz\mag'
# voice_dir = r'E:\MAG\code_final_version\data\test_subject\voice'
#
# sensor_data_sampling_rate = 500
# for file_id in range(11,21):
#     name = str(file_id)+'.csv'
#     mag_filename = os.path.join(mag_dir,name)
#     mag_save_filename = os.path.join(mag_save_dir,name)
#     voice_filename = os.path.join(voice_dir,str(file_id)+'.wav')
#
#     csv_to_1000Hz(voice_filename, mag_filename, mag_save_filename)

# # -----------------in_the_wild-----------------
# import os
# mag_dir = r'E:\MAG\code_final_version\data\in_the_wild\aligned\mag'
# mag_save_dir = r'E:\MAG\code_final_version\data\in_the_wild\aligned_1000Hz\mag'
# voice_dir = r'E:\MAG\code_final_version\data\in_the_wild\voice'
# names = ['DR1', 'DR3']
# sensor_data_sampling_rate = 500
#
# for file_id in names:
#     name = str(file_id)+'.csv'
#     mag_filename = os.path.join(mag_dir,name)
#     mag_save_filename = os.path.join(mag_save_dir,str(file_id)+'.csv')
#     voice_filename = os.path.join(voice_dir,str(file_id)+'.wav')
#
#     csv_to_1000Hz(voice_filename, mag_filename, mag_save_filename)

# -----------------huawei_table_lab-----------------
import os
mag_dir = r'E:\MAG\code_final_version\data\huawei_table_lab\aligned\mag'
mag_save_dir = r'E:\MAG\code_final_version\data\huawei_table_lab\aligned_1000Hz\mag'
voice_dir = r'E:\MAG\code_final_version\data\huawei_table_lab\voice'

sensor_data_sampling_rate = 500
for file_id in range(1,11):
    name = str(file_id)+'.csv'
    wav_name = "%02d" % file_id
    mag_filename = os.path.join(mag_dir,name)
    mag_save_filename = os.path.join(mag_save_dir,"%02d.csv" % file_id)
    voice_filename = os.path.join(voice_dir,wav_name+'.wav')

    csv_to_1000Hz(voice_filename, mag_filename, mag_save_filename)