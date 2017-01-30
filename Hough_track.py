import os
import sys
import time

import movie_instance as mi
import tracking_helper_functions as thf
from settings import tracking_settings

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

    if 'output_dir' in tracking_settings:
        output_dir = tracking_settings['output_dir']
    else:
        output_dir = './'

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for fn in fns:
        movie = mi.GyroMovieInstance(fn)
        output = fn.split('/')[-1]
        output = output.split('.')[0]
        if not os.path.exists(output):
            os.mkdir(output)

        checks = output + '/checks/'
        if not os.path.exists(checks):
            os.mkdir(checks)

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
        for i in xrange(lf - ff):

            ind = i + ff
            movie.extract_frame_data(ind)
            movie.adjust_frame()

            movie.center_on_bright(5)

            if 'tracked_image' in tracking_settings:
                movie.save_frame_with_boxes(name=output + '/' + '%03d' % ind)

            if i % 100 == 0:
                et = time.time()
                if i > 0:
                    print 'frame', i, 'tracked... ... %0.2f s per frame' % ((et - st) / i)
                    if i % 200 == 0:
                        movie.save_frame_with_boxes(name=checks + '/' + '%03d' % ind)

            if ind in tracking_settings['cf']:
                movie.find_points_hough()
                movie.save_frame_with_boxes(name=output + '/' + '%03d' % ind)
            if i == (lf - ff) - 1:
                movie.save_frame_with_boxes(name=output + '/' + '%03d' % ind)
