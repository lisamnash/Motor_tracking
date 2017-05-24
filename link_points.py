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
from scipy.signal import savgol_filter
from video_analyzing.new_mode_functions import load_linked_data_and_window
import sys


def link_points(root_dir):
    file = os.path.join(root_dir, 'steps/steps.hdf5')

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

        single_data[:, 0] = savgol_filter(single_data[:, 0], 7, 1)
        single_data[:, 1] = savgol_filter(single_data[:, 1], 7, 1)
        key_name = '%03d' % i

        dset = new.create_dataset((key_name), np.shape(single_data), dtype='float', data=single_data)

        image_path = os.path.join(root_dir, 'gyro_path_images/')
        if not os.path.exists(image_path):
            os.mkdir(image_path)

        image_path = os.path.join(image_path, '%03d.png' % i)
        # fig = plt.figure()
        # len_single = len(single_data)
        # plt.plot(times[:len_single], single_data[:, 0])
        # plt.savefig(image_path)
        # plt.close()

    dset = new.create_dataset('times', np.shape(times), dtype='float', data=times)
    new.close()
    data.close()


def filter_by_frequency(data_path, frequency):
    data = h5py.File(data_path, 'r')
    keys = data.keys()

    # for key in keys:


if __name__ == '__main__':
    root_dir = '/Volumes/labshared2/Lisa/2017_02_21/tracked/7p_0p0A_5p5A_1_2/'

    path = '/Volumes/GetIt/saved_stuff/2017_05_18/1p77_1_2017_05_18/'
    # [np.array(x), np.array(y), np.array(x_mean), np.array(y_mean), np.array(time)]

    x, y, x_mean, y_mean, time = load_linked_data_and_window(path + 'com_data.hdf5', window=False)

    fft_freq = np.fft.fftfreq(len(x[0]), time[1] - time[0])

    diff = np.abs(1.77 - fft_freq)
    closest = np.where(diff == np.min(diff))[0][0]
    closest_freq = fft_freq[closest]
    print closest_freq

    fft_freq_delta = fft_freq[1] - fft_freq[0]
    print fft_freq_delta
    num = np.floor(0.15 / fft_freq_delta)

    if num % 2 == 0:
        num += 1

    window = np.hanning(num)

    half = np.floor(num / 2.)

    closest_adj = closest - half
    big_window = np.zeros_like(x[0])
    big_window[closest_adj:closest_adj + num] = window

    closest_neg = np.where(fft_freq == -closest_freq)[0][0]
    closest_adj = closest_neg - half

    closest_adj = closest - half
    big_window = np.zeros_like(x[0])
    big_window[closest_adj:closest_adj + num] = window

    path = os.path.join(path, 'com_data_filtered_0.1.hdf5')
    new_ds = h5py.File(path, "w")

    for i in xrange(len(x)):
        this_x = x[i]
        this_y = y[i]


        ff = np.fft.fft(this_x + 1j * this_y)

        print i
        # fig = plt.figure()
        # plt.plot(fft_freq, big_window * np.abs(ff)**2)
        # plt.show()

        new = np.fft.ifft(big_window * ff)

        new_x = np.real(new)
        new_y = np.imag(new)

        # fig = plt.figure()
        # plt.plot(time, np.real(new))
        # plt.plot(time, np.imag(new))
        # plt.show()

        key_name = '%03d' % i

        #fig = plt.figure()
        #plt.plot(new_x)
        #plt.plot(this_x, 'ro', alpha = 0.3)
        #plt.show()

        single_data = np.array([x_mean[i] + new_x, y_mean[i] + new_y]).T

        dset = new_ds.create_dataset((key_name), np.shape(single_data), dtype='float', data=single_data)
    dset = new_ds.create_dataset('times', np.shape(time), dtype='float', data=time)
    new_ds.close()