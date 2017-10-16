import numpy as np
import os
import pandas as pd



data_dir = '/scratch/kvg245/tsing_data/Formated_Data/Experiment_2/'


def get_angles(row):
    x = row['UnitQuaternion.x']
    y = row['UnitQuaternion.y']
    z = row['UnitQuaternion.z']
    w = row['UnitQuaternion.w']
    roll = np.arctan2(2*(x*y+z*w),1-2*(y**2+z**2))
    pitch = np.arcsin(2*(x*z-y*w))
    yaw = np.arctan2(2*(x*w-z*y),1-2*(z**2+w**2))
    x = np.cos(yaw)*np.cos(pitch)
    y = np.sin(yaw)*np.cos(pitch)
    z = np.sin(pitch)
    theta = np.arctan2(y,x)
    phi = np.arctan2((x**2+y**2)**0.5,z)
    return theta,phi

video_meta = pd.read_csv(data_dir+'videoMeta.csv')
worker_list = [x for x in os.listdir(data_dir) if 'video' not in x]
data_arr = []
for w, worker in enumerate(worker_list):
    data_arr.append([])
    video_list = os.listdir(data_dir+worker)
    for i,video in enumerate(video_list):
        data = pd.read_csv(data_dir+worker+'/'+video)
        wxp_data = []
        print w,i
        for r, row in data.iterrows():
            if r>5:
                theta,phi = get_angles(row)
                time = row['PlaybackTime']
                frame_num = int(float(time)*
                        float(video_meta.iloc[[i]]['FrameRate']))
                wxp_data.append([frame_num,time,theta,phi])
        data_arr[w].append(wxp_data)
data_array = np.array(data_arr)
np.save(data_dir+'data1.npy',data_array)
