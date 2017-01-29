import argparse
import sys

import movie_instance as mi

if __name__ == '__main__':
    '''example usage python Hough_track.py -i /Users/lisa/Dropbox/research/lisa/Motor_tracking_code/test_cine/2_Cam_7326_Cine3.cine -min_radius 25 -max_radius 33 -o /Users/lisa/Dropbox/research/lisa/Motor_tracking_code -min_val 0.02 -max_val 0.4 -pix 4'''

    parser = argparse.ArgumentParser(description='Track points from a video')

    parser.add_argument('-i', dest='input_file', type=str, help='input cine file. required', required=True)
    parser.add_argument('-o', dest='output_directory', type=str, help='output directory.', default='./')
    parser.add_argument('-ff', dest='ff', type=int, help='first frame to track', default=0)
    parser.add_argument('-lf', dest='lf', type=int, help='last frame to track', default=-1)
    parser.add_argument('-append_fn', type=str, help='name to append to output directory for descriptive purposes')
    parser.add_argument('-min_val', type=float, help='minimum brightness', default=0.05)
    parser.add_argument('-max_val', type=float, help='maximum brightness', default=0.7)

    parser.add_argument('-pix', type=int, help="Length of tracking box", default=8)
    parser.add_argument('-cf', dest='cf', nargs='+', type=int,
                        help='Frames to perform convolution. Use this if the point gets lost in a frame.', default=[0])

    parser.add_argument('-min_radius', type=int, help='Minimum radius for Hough transform', default=17)
    parser.add_argument('-max_radius', type=int, help='Maximum radius for Hough transform', default=23)
    parser.add_argument('-tracked_image', type=bool, help='If True, plots all the frames with tracked points.',
                        default=False)
    parser.add_argument('-save_during', type=bool,
                        help='If True, saves the points at each time step in a folder.  Does not save com_data file',
                        default=False)
    parser.add_argument('-save_com', type=bool, help='If True, saves com_data file', default=True)
    parser.add_argument('-skip_Hough', dest='skip_Hough', type=bool,
                        help='If True, skips Hough transform on all but the first frame', default=False)
    parser.add_argument('-ffE', type=int, help='frames from end?', default=-1)

    args = parser.parse_args()

    fns = []
    if args.input_file:
        fn = args.input_file

        spl = fn.split('.')[-1]

        if spl == 'cine':
            fns = [fn]
        else:
            # find all cines in directory
            fns = mtf.find_files_by_extension(fn, '.cine', tot=True)


    else:
        print 'No input file selected.  Exiting'
        sys.exit()

    tracked_image = args.tracked_image
    for fn in fns:
        movie = mi.GyroMovieInstance(fn)
        movie.set_tracking_size(args.pix)
        movie.min_radius = args.min_radius
        movie.max_radius = args.max_radius
        movie.set_min_max_val(args.min_val, args.max_val)

        ff = args.ff

        if args.lf == -1:
            lf = movie.num_frames
        else:
            lf = args.lf

        for i in xrange(lf - ff):
            ind = i + ff
            movie.extract_frame_data(ind)
            movie.adjust_frame()

            if ind in args.cf or i == (lf - ff) - 1:
                movie.find_points_hough()
                movie.save_frame_with_boxes(name='%03d' % ind)

            movie.center_on_bright(5)

            if tracked_image:
                movie.save_frame_with_boxes(name='%03d' % ind)

            if i % 100 == 0:
                print 'frame', i , 'tracked...'
