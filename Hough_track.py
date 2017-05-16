import h5py
import link_points as lp
import matplotlib.image as mpimg
import movie_instance as mi
import numpy as np
import os
import sys
import time
import tracking_helper_functions as thf
import Experiment_movies
from settings import tracking_settings


def run_tracking(tracking_settings):
    fns = []

    if 'input_dir' in tracking_settings:
        fn = tracking_settings['input_dir']

        spl = fn.split('.')[-1]

        if spl == 'cine':
            fns = [fn]
        else:
            # find all cines in directory
            fns = thf.find_files_by_extension(fn, '.cine', tot=True)

            if len(fns) < 1:
                fns = [fn]

    else:
        print 'No input file selected.  Exiting'
        sys.exit()

    output_dir = thf.set_output_directory()

    outputs = []
    for fn in fns:

        if True:  # try:
            movie = mi.GyroMovieInstance(fn)
            output = fn.split('/')[-1]
            output = output_dir + '/' + output.split('.')[0]
            outputs.append(output)
            if not os.path.exists(output):
                os.mkdir(output)
            else:
                output = output
                if not os.path.exists(output):
                    os.mkdir(output)
            if not os.path.exists(os.path.join(output, 'com_data.hdf5')):
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

                    if (i in tracking_settings['cf']) or ('all' in tracking_settings['cf']):
                        movie.find_points_hough()
                        # movie.find_points_convolution()
                        movie.save_frame_with_boxes(name=output + '/' + '%03d' % ind)



                    movie.center_on_bright(2)

                    if 'tracked_image' in tracking_settings:
                        movie.save_frame_with_boxes(name=output + '/' + '%03d' % ind)

                    if i % 100 == 0:
                        et = time.time()
                        if i >= 0:
                            print 'frame', i, 'tracked... ... %0.2f s per frame' % ((et - st) / (i + 1))
                            if i % 100 == 0:
                                movie.save_frame_with_boxes(name=checks + '/' + '%03d' % ind)

                    if (i in tracking_settings['cf']):  # or ('all' in tracking_settings['cf']):

                        movie.save_frame(name=output + '/' + '%03d_nb' % ind)

                    if i == (lf - ff) - 1:
                        movie.save_frame_with_boxes(name=output + '/' + '%03d' % ind)

                    t = np.array(
                        [[movie.current_time, movie.frame_current_points[j, 0], movie.frame_current_points[j, 1]] for j
                         in
                         range(len(movie.frame_current_points))])

                    com_data.append(t)
                    path = os.path.join(path_to_step_data, 'steps.hdf5')

                    if i == 0:
                        f = h5py.File(path, "w")

                    dset = f.create_dataset(('step_%05d' % ind), np.shape(t), dtype='float', data=t)

                f.close()

                lp.link_points(output)
        else:  # except:
            print 'error'

    return fns, outputs
