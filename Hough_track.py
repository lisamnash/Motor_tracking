import os
import sys
import time

import numpy as np

import movie_instance as mi
import tracking_helper_functions as thf
from settings import tracking_settings

import pandas as pd
import json
import h5py
import matplotlib.image as mpimg

if __name__ == '__main__':

    fns = []
    if 'input_dir' in tracking_settings:
        fn = tracking_settings['input_dir']

        spl = fn.split('.')[-1]

        if spl == 'cine':
            fns = [fn]
        else:
            # find all cines in directory
            fns = thf.find_files_by_extension(fn, '.cine', tot=True)

    else:
        print 'No input file selected.  Exiting'
        sys.exit()

    output_dir = thf.set_output_directory()

    for fn in fns:
        movie = mi.GyroMovieInstance(fn)
        output = fn.split('/')[-1]
        output = output_dir + '/' + output.split('.')[0]
        if not os.path.exists(output):
            os.mkdir(output)
        else:
            output = output + '_2'
            if not os.path.exists(output):
                os.mkdir(output)

        checks = os.path.join(output, 'checks')
        if not os.path.exists(checks):
            os.mkdir(checks)

        # for saving steps
        path_to_step_data = os.path.join(output, 'steps')
        if not os.path.exists(path_to_step_data): os.mkdir(path_to_step_data)

        if 'pix' in tracking_settings:
            movie.set_tracking_size(tracking_settings['pix'])

        if 'min_max_radius' in tracking_settings:
            movie.min_radius = tracking_settings['min_max_radius'][0]
            movie.max_radius = tracking_settings['min_max_radius'][1]

        if 'min_max_val' in tracking_settings:
            movie.set_min_max_val(tracking_settings['min_max_val'][0], tracking_settings['min_max_val'][1])

        if 'first_frame' in tracking_settings:
            ff = tracking_settings['first_frame']
        else:
            ff = 0

        if 'last_frame' in tracking_settings:
            if tracking_settings['last_frame'] >= ff:
                lf = tracking_settings['last_frame']
            else:
                lf = movie.num_frames
        else:
            lf = movie.num_frames

        st = time.time()
        com_data = []
        for i in xrange(lf - ff):

            ind = i + ff
            movie.extract_frame_data(ind)
            movie.adjust_frame()

            if 'tracked_image' in tracking_settings:
                movie.save_frame_with_boxes(name=output + '/' + '%03d' % ind)

            if i % 100 == 0:
                et = time.time()
                if i > 0:
                    print 'frame', i, 'tracked... ... %0.2f s per frame' % ((et - st) / i)
                    if i % 100 == 0:
                        movie.save_frame_with_boxes(name=checks + '/' + '%03d' % ind)

            if (i in tracking_settings['cf']) or ('all' in tracking_settings['cf']):
                movie.find_points_hough()


            movie.center_on_bright(4)

            if (i in tracking_settings['cf']):  # or ('all' in tracking_settings['cf']):
                movie.save_frame_with_boxes(name=output + '/' + '%03d' % ind)
                movie.save_frame(name=output + '/' + '%03d_nb' % ind)

            if i == (lf - ff) - 1:
                movie.save_frame_with_boxes(name=output + '/' + '%03d' % ind)

            t = np.array(
                [[movie.current_time, movie.frame_current_points[j, 0], movie.frame_current_points[j, 1]] for j in
                 range(len(movie.frame_current_points))])

            com_data.append(t)

            df = pd.DataFrame(data=t)
            # path = os.path.join(path_to_step_data, 'step_%05d.csv' % i)

            # df.to_csv(path, header=True, index=False)

            path = os.path.join(path_to_step_data, 'steps.json')
            path = os.path.join(path_to_step_data, 'steps.hdf5')

            if i == 0:
                f = h5py.File(path, "w")

            dset = f.create_dataset(('step_%05d' % ind), np.shape(t), dtype='float', data=t)

            # f_dict = {
            #     ('step_%05d' % ind): ({'t': movie.current_time, 'pts_x': movie.frame_current_points[:, 0].tolist(),
            #                            'pts_y': movie.frame_current_points[:, 0].tolist()})}
            #
            # if i > 0:
            #     config = json.loads(open(path).read())
            #     config.update(f_dict)
            #     with open(path, 'w') as f:
            #         f.write(json.dumps(config))
            # else:
            #
            #     with open(path, 'w') as f:
            #         f.write(json.dumps(f_dict))
        f.close()
        com_data = np.array(com_data)
        sh1, sh2, sh3 = np.shape(com_data)
        com_data = com_data.flatten()
        com_data = np.reshape(com_data, [sh1 * sh2, 3])

        thf.dump_pickled_data(output, 'com_data', com_data)

        print 'com data', com_data

        thf.fft_on_data(np.array(com_data), output_dir=output + '/')
