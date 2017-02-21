import glob
import h5py
import matplotlib.pyplot as plt
import movie_instance as mi
import numpy as np
import os
import pandas as pd
import seaborn
import sys
import time
import tracking_helper_functions as thf
from settings import tracking_settings


def moving_average(values, window):
    vv = values.T

    new = []
    for i in xrange(len(vv)):
        weigths = np.repeat(1.0, window) / window
        v = np.convolve(vv[i], weigths, 'valid')
        new.append(v)
    new = np.array(new)
    return new.T


if __name__ == '__main__':
    root_dir = '/Volumes/labshared2/Lisa/2017_02_20_different_lighting/tracked/7p_6A_1_2/'

    file = os.path.join(root_dir, 'steps/steps.hdf5')
    num = 54

    data = h5py.File(file, 'r')
    keys = data.keys()

    for i in xrange(len(keys)):
        key = keys[i]
        dat = len(np.array(data[key]))
        if dat == 54:
            start_key = key
            break
        else:
            start_key = keys[0]



    compare = np.array(data[start_key])

    fig = plt.figure()
    plt.scatter(compare[:, 1], compare[:, 2], c=range(len(compare)), cmap=plt.cm.coolwarm)
    plt.gca().set_aspect(1)
    plt.savefig(os.path.join(root_dir, 'color_by_number.png'))

    path = os.path.join(root_dir, 'com_data.hdf5')
    new = h5py.File(path, "w")

    com_data = []
    for i in xrange(len(compare)):
        single_data = []
        times = []
        count = 0
        pt = compare[i]
        for key in keys:
            step_data = np.array(data[key])
            t = step_data[0, 0]
            dist = np.sum(((step_data - pt) ** 2)[:, 1:], axis=1)

            ind = np.where(dist < 15)[0]
            if len(ind) > 0:
                times.append(t)
                count += 1
                single_data.append(step_data[ind][0, 1:])
                pt = step_data[ind][0]
            else:
                times.append(t)
                count += 1
                single_data.append(pt[1:])

        single_data = np.array(single_data)
        single_data = moving_average(single_data, 6)
        key_name = '%03d' % i

        dset = new.create_dataset((key_name), np.shape(single_data), dtype='float', data=single_data)

        image_path = os.path.join(root_dir, 'gyro_path_images/')
        if not os.path.exists(image_path):
            os.mkdir(image_path)

        image_path = os.path.join(image_path, '%03d.png'%i)
        fig = plt.figure()
        len_single = len(single_data)
        plt.plot(times[:len_single], single_data[:,0])
        plt.savefig(image_path)

    dset = new.create_dataset('times', np.shape(times), dtype='float', data=times)
    new.close()
    data.close()
