from h import *
import pandas as pd


def read_mag(mag_data):
    mag = []
    for i in mag_data.split(" ")[:-1]:  # 去掉最后一个回车
        if len(i) > 0:
            mag.append(to_12bit_int(i))
    return mag


def magtxt_to_json(magtxt_filename):
    ans_list = []
    with open(magtxt_filename, 'r') as f:
        raw_data = f.read()
        data_list = raw_data.split('\n')[:-1]
        for idx, i in enumerate(data_list):
            if idx % 3 == 1:
                mag = read_mag(i)
                ans_list.append(mag)
            if idx % 3 == 2:  # 补上磁力计没有采集数据的时间
                time = int(i.split(" ")[-1])/1e6
                ans_list.append([ans_list[-1][-1] for j in range(int(time*500))])
    mag = np.hstack([i for i in ans_list])
    print("mag len",len(mag)/500)
    return mag


def save_csv(save_filename, mag_data):
    tmp_zeros = np.zeros(mag_data.shape)
    save_csv_df = pd.DataFrame({'x': tmp_zeros, 'y': tmp_zeros, 'z': mag_data})
    save_csv_df.to_csv(save_filename, index=False, sep=',')
    print("save csv", save_filename)

# # -----------------robustness-----------------
# name = 'xiaomi'
# magtxt_filename = 'E:\\MAG\\code_final_version\\data\\robustness\\' + name + '\\raw_data\mag\\mag.txt'
# save_filename = 'E:\\MAG\\code_final_version\\data\\robustness\\' + name + '\\raw_data\\mag_csv\\mag.csv'
#
# mag_data = magtxt_to_json(magtxt_filename)
# save_csv(save_filename, mag_data)

# # -----------------test_subject-----------------
# import os
# mag_dir = r'E:\MAG\code_final_version\data\test_subject\raw_data\mag'
#
# mag_save_dir = r'E:\MAG\code_final_version\data\test_subject\raw_data\mag_csv'
#
# voice_dir = r'E:\MAG\code_final_version\data\test_subject\voice'
#
# sensor_data_sampling_rate = 500
# for file_id in range(11,21):
#     name = str(file_id)+'.txt'
#     mag_filename = os.path.join(mag_dir,name)
#
#     mag_save_filename = os.path.join(mag_save_dir,str(file_id)+'.csv')
#
#     voice_filename = os.path.join(voice_dir,str(file_id)+'.wav')
#
#     mag_data = magtxt_to_json(mag_filename)
#     save_csv(mag_save_filename, mag_data)

# # -----------------in_the_wild-----------------
# import os
# mag_dir = r'E:\MAG\code_final_version\data\in_the_wild\raw_data\mag'
#
# mag_save_dir = r'E:\MAG\code_final_version\data\in_the_wild\raw_data\mag_csv'
#
# voice_dir = r'E:\MAG\code_final_version\data\in_the_wild\voice'
# names = ['DR1', 'DR3']
# sensor_data_sampling_rate = 500
# for file_id in names:
#     name = str(file_id)+'.txt'
#     mag_filename = os.path.join(mag_dir,name)
#
#     mag_save_filename = os.path.join(mag_save_dir,str(file_id)+'.csv')
#
#     voice_filename = os.path.join(voice_dir,str(file_id)+'.wav')
#
#     mag_data = magtxt_to_json(mag_filename)
#     save_csv(mag_save_filename, mag_data)

# -----------------huawei_table_lab-----------------
import os
mag_dir = r'E:\MAG\code_final_version\data\huawei_table_lab\raw_data\mag'
mag_save_dir = r'E:\MAG\code_final_version\data\huawei_table_lab\raw_data\mag_csv'
voice_dir = r'E:\MAG\code_final_version\data\huawei_table_lab\voice'

sensor_data_sampling_rate = 500
for file_id in range(1,11):
    name = str(file_id)+'.txt'
    wav_name = "%02d" % file_id
    mag_filename = os.path.join(mag_dir,name)

    mag_save_filename = os.path.join(mag_save_dir,str(file_id)+'.csv')

    voice_filename = os.path.join(voice_dir,wav_name+'.wav')

    mag_data = magtxt_to_json(mag_filename)
    save_csv(mag_save_filename, mag_data)