import Hough_track as ht
import Experiment_movies.movie_new_format as mnf
import time

if __name__=='__main__':
    print 'running pipeline...'
    start_time = time.time()
    end_time = time.time()
    while end_time-start_time < 3600:
        from settings import tracking_settings
        print 'tracking...'
        fns, outputs = ht.run_tracking(tracking_settings)
        print 'done tracking...'
        for i in xrange(len(outputs)):
            fn_for_movie = outputs[i].split('/')[-1] + '.mp4'
            print fn_for_movie
            mnf.make_frames(outputs[i], fns[i], color_by = 'phase', cmap = 'isolum_rainbow')


