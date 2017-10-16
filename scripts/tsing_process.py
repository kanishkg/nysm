import numpy as np
import os
import pandas as pd



data_dir = '/sratch/kvg245/tsing_data/Formated_Data/Experiment_1/'


def get_angles(row):
    x = row['UnitQuarternion.x']
    y = row['UnitQuarternion.y']
    z = row['UnitQuarternion.z']
    w = row['UnitQuarternion.w']
    roll = np.arctan2(2*(x*y+z*w),1-2*(y**2+z**2))
    pitch = np.arcsin(2*(x*z-y*w))
    yaw = np.arctan2(2*(x*w-z*y),1-2(z**2+w**2))
    x = np.cos(yaw)*np.cos(pitch)
    y = np.sin(yaw)*np.cos(pitch)
    z = np.sin(pitch)
    theta = np.arctan2(y,x)
    phi = np.arctan2((x**2,y**2)**0.5,z)
    return theta,phi

video_meta = pd.read_csv(data_dir+'videoMeta.csv')
worker_list = os.listdir(data_dir)
data_arr = []
for worker,w in enumerate(worker_list):
    data_arr.append([])
    video_list = os.listdir(data_dir+worker)
    for video,i in enumerate(video_list):
        data = pd.read_csv(data_dir+worker+'/'+video)
        wxp_data = []
        frames_done =[]
        for row,r in enumerate(data.rows):
            if r>5:
                theta,phi = get_angles(row)
                time = row['PlabackTime']
                frame_num = int(float(time)*
                        float(video_meta.iloc[[i]]['FrameRate']))
                if frame_num not in frames_done:
                    wxp_data.append([frame_num,time,theta,phi])
                frames_done.append(frame_num)
        data_arr[w].append(wxp_data)
data_array = np.array(data_arr)

