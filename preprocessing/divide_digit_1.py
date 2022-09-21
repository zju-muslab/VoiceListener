from h import *
import pandas as pd
import os

def read_csv_z_axis(sensor_filename):
    df = pd.read_csv(sensor_filename, encoding='utf-8')
    z_axis = list(df["z"])
    return z_axis

def read_csv_xyz_axis(sensor_filename):
    df = pd.read_csv(sensor_filename, encoding='utf-8')
    x_axis = list(df["x"])
    y_axis = list(df["y"])
    z_axis = list(df["z"])
    return np.vstack((x_axis,y_axis,z_axis))

def read_json(filename):
    with open(filename, 'r', encoding='utf8')as fp:
        json_data = json.load(fp)
        return json_data


# -----------------huawei_table_lab-----------------
mag_dir = r'E:\MAG\code_final_version\data\huawei_table_lab\aligned\mag'
mag_save_dir = r'E:\MAG\code_final_version\data\huawei_table_lab\digit\sensor\mag'

acc_dir = r'E:\MAG\code_final_version\data\huawei_table_lab\aligned\acc'
acc_save_dir = r'E:\MAG\code_final_version\data\huawei_table_lab\digit\sensor\acc'

gyr_dir = r'E:\MAG\code_final_version\data\huawei_table_lab\aligned\gyr'
gyr_save_dir = r'E:\MAG\code_final_version\data\huawei_table_lab\digit\sensor\gyr'

voice_dir = r'E:\MAG\code_final_version\data\huawei_table_lab\voice'
voice_save_dir = r'E:\MAG\code_final_version\data\huawei_table_lab\digit\voice'

label_dir = r'E:\MAG\code_final_version\data\huawei_table_lab\labels'

down_sampling_rate = 500

# 初始话要保存的数据
new_vocie = {}
new_acc = {}
new_gyr = {}
new_mag = {}
for i in range(10):
    new_vocie[i] = []
    new_acc[i] = []
    new_gyr[i] = []
    new_mag[i] = []

for file_id in range(1,11):
    name = str(file_id)+'.csv'
    wav_name = "%02d" % file_id

    mag_filename = os.path.join(mag_dir,name)

    acc_filename = os.path.join(acc_dir,name)

    gyr_filename = os.path.join(gyr_dir,name)

    voice_filename = os.path.join(voice_dir,wav_name+'.wav')

    # read label
    label_filename = os.path.join(label_dir,str(file_id)+'.json')
    label = read_json(label_filename)["labels"]
    label = [i for i in label if i != '-1']
    # read wav
    sampling_rate, voice = read_wav(voice_filename)
    # voice = voice.copy()[::sampling_rate//down_sampling_rate]
    # read mag
    mag_xyz = read_csv_xyz_axis(mag_filename)
    # read acc
    acc_xyz = read_csv_xyz_axis(acc_filename)
    # read gyr
    gyr_xyz = read_csv_xyz_axis(gyr_filename)


    idxs = []
    voice_idx = 0
    voice_list = []
    acc_list = []
    gyr_list = []
    mag_list = []

    while voice_idx<len(voice):
        if np.abs(float(voice[voice_idx]))<0.0005:
            # print(float(voice[voice_idx]),voice_idx)
            cnt = 0
            begin_voice_idx = voice_idx
            while voice_idx+1 < len(voice) and voice[voice_idx+1]<0.0005:
                cnt += 1
                voice_idx += 1
            if cnt > 40*96:
                idxs.append(begin_voice_idx)
                idxs.append(voice_idx-1)
        voice_idx+=1
    MM = 1E9
    # 生成按数字切分后的voice acc gyr mag数据
    for id, i in enumerate(idxs):
        if id ==len(idxs)-1:
            break
        if np.sum(np.abs(voice[idxs[id]:idxs[id+1]]))>2e6 and idxs[id+1]-idxs[id]>150*96:
            if mag_xyz.shape[1]<idxs[id+1]/96: # 因为3号数据磁力计缺了一段
                break
            if np.sum(np.abs(voice[idxs[id]:idxs[id+1]]))<MM:
                MM = np.sum(np.abs(voice[idxs[id]:idxs[id+1]]))
            voice_list.append(voice[idxs[id]:idxs[id+1]])
            acc_list.append(acc_xyz[:, idxs[id]//96:idxs[id+1]//96])
            gyr_list.append(gyr_xyz[:, idxs[id]//96:idxs[id + 1]//96])
            mag_list.append(mag_xyz[:, idxs[id]//96:idxs[id + 1]//96])
            # 展示切割出来的每一段数据
            # plt.figure()
            # plt.subplot(4,1,1)
            # plt.plot(voice_list[-1][::48000//500])
            # plt.subplot(4,1,2)
            # plt.plot(acc_list[-1][2,:])
            # plt.subplot(4,1,3)
            # plt.plot(gyr_list[-1][2,:])
            # plt.subplot(4,1,4)
            # plt.plot(mag_list[-1][2,:])
            # plt.show()
    print(MM)
    print("voice list length",len(voice_list))

    # 重构0到9的数据
    for digit_idx,i in enumerate(label):
        if digit_idx<len(voice_list): # 因为3号数据磁力计缺了一段
            new_vocie[int(i)].append(voice_list[digit_idx])
            new_acc[int(i)].append(acc_list[digit_idx])
            new_gyr[int(i)].append(gyr_list[digit_idx])
            new_mag[int(i)].append(mag_list[digit_idx])

for digit in range(10):
    digit_voice = np.hstack(new_vocie[digit])
    digit_acc = np.hstack(new_acc[digit])
    digit_gyr = np.hstack(new_gyr[digit])
    digit_mag = np.hstack(new_mag[digit])

    plt.figure()
    plt.plot(digit_voice)
    plt.title(digit)
    plt.show()

    # 保存文件的地址
    acc_save_filename = os.path.join(acc_save_dir, str(digit)+'.csv')
    mag_save_filename = os.path.join(mag_save_dir, str(digit)+'.csv')
    gyr_save_filename = os.path.join(gyr_save_dir, str(digit) + '.csv')
    voice_save_filename = os.path.join(voice_save_dir, str(digit) + '.wav')
    # voice_save_filename = os.path.join(voice_save_dir,str(file_id),str(digit)+'.wav')
    # acc_save_filename = os.path.join(acc_save_dir,str(file_id),str(digit)+'.csv')
    # gyr_save_filename = os.path.join(gyr_save_dir, str(file_id), str(digit) + '.csv')
    # mag_save_filename = os.path.join(mag_save_dir, str(file_id), str(digit) + '.csv')

    # print("save wav ",voice_save_filename,"digit num",len(new_vocie[digit]))

    # if not os.path.exists(os.path.join(voice_save_dir,str(file_id))):
    #     os.makedirs(os.path.join(voice_save_dir,str(file_id)))
    # if not os.path.exists(os.path.join(acc_save_dir,str(file_id))):
    #     os.makedirs(os.path.join(acc_save_dir,str(file_id)))
    # if not os.path.exists(os.path.join(gyr_save_dir,str(file_id))):
    #     os.makedirs(os.path.join(gyr_save_dir,str(file_id)))
    # if not os.path.exists(os.path.join(mag_save_dir,str(file_id))):
    #     os.makedirs(os.path.join(mag_save_dir,str(file_id)))



    save_wav(voice_save_filename,sampling_rate,digit_voice)
    acc_save_csv_df = pd.DataFrame({'x': digit_acc[0,:], 'y': digit_acc[1,:], 'z': digit_acc[2,:]})
    acc_save_csv_df.to_csv(acc_save_filename, index=False, sep=',')

    gyr_save_csv_df = pd.DataFrame({'x': digit_gyr[0,:], 'y': digit_gyr[1,:], 'z': digit_gyr[2,:]})
    gyr_save_csv_df.to_csv(gyr_save_filename, index=False, sep=',')

    mag_save_csv_df = pd.DataFrame({'x': digit_mag[0,:], 'y': digit_mag[1,:], 'z': digit_mag[2,:]})
    mag_save_csv_df.to_csv(mag_save_filename, index=False, sep=',')





