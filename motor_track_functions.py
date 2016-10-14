import scipy.ndimage.filters as filters
import scipy.ndimage as ndimage
import cPickle as pickle
import math
import itertools
import time
import pylab as P
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
from numpy import *
import cine
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import os
import scipy.spatial as spatial
import numpy as np
import cv2

from skimage import data, color
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter

#functions for tracking motors

def load_pickled_data(root_dir, filename = -1):
    '''loads data from pickle file.

    Parameters
    ----------
    root_dir : string
        Directory in which file is saved
    filename : string 
        name of file

    Returns
    ----------
    data : any python object
        data from pickled file
        '''
    if filename ==-1:
        tot_path = root_dir
    else:
        tot_path = root_dir + '/' + filename

    try :
        of = open(tot_path, 'rb')
        data = pickle.load(of)
    except Exception:
        data = 0
        print 'file not found' 
    return data



def find_circles(img, fs, **kwargs):
    '''
    Finds circles in an image using a Hough transform.  If frame_number > 0, uses a kdtree to order points and match between frames.
    
    Parameters
        -----------------
        img: float array
            Array of values making up the image of one frame
        fs: float array (1D x Num points)
            The tracked points in the previous frame
            
    
        Returns
        ---------------
        points: float array
            list of tracked points in frame
        circles:
            circles (with radii) returned by Hough transform
        img: float array
            The frame after blur has been applied
        flag: boolean
            tells whether the correct number (same as previous frame) number of point was found in frame
        
        '''
    #convert image to uint8
    img = np.array(img*255, dtype =uint8)
    
    #apply blur so you don't find lots of fake circles
    img = cv2.GaussianBlur(img, (5,5), 2, 2)
    
    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    cimg = cimg
    
    if 'min_rad' in kwargs:
        min_rad = kwargs['min_rad']
    else:
        min_rad = 15
    
    if 'max_rad' in kwargs:
        max_rad = kwargs['max_rad']
    else:
        max_rad = 35
   
   
    circles = cv2.HoughCircles(img,cv2.cv.CV_HOUGH_GRADIENT,1,20,
                          param1=48,param2=18,minRadius=min_rad,maxRadius=max_rad)

    circles = np.uint16(np.around(circles))
    circles = circles[0]
    points = array([circles[:,0], circles[:,1]]).T
    
    #if order is in kwargs then this is not the first frame.  Points need to be linked up between frames.
    if 'order' in kwargs:
        order = kwargs['order']
    else:
        order = False
    
    fs_flat = np.array(fs).flatten()
    fz = fs_flat[0]

    points = np.array(points, dtype = float)
    if fz != -1:

        if len(points) == len(fs): 
        
        
            if order:
                dist, indexes = do_kdtree(fs, points)
           
                points = points[indexes]
                circles = circles[indexes]
            flag = True
        else:
            dist, indexes = do_kdtree(points, points, k=2)
            dist = dist[:,1]
     
            
            
            outliers = np.where(dist>1.2*mean(dist.flatten()))[0]
            
            if len(outliers)>0:

                points = np.delete(points, outliers, 0)
                circles = np.delete(circles, outliers, 0)


            if len(points) == len(fs): 
        
        
                if order:
                    dist, indexes = do_kdtree(fs, points)
           
                    points = points[indexes]
                    circles = circles[indexes]
                flag = True
                
              
            else:
                points = fs
                flag = False
       
    else:
        flag = True
        
    points = do_center(img, points)

    
    
    return points, np.array([circles]), img, flag

def do_center(frame, points, pix=4):
    '''centers a box around white dot found with Hough transform.

    Parameters
    ----------
    frame : float array
        image array
    points : float array 
        x and y coordinates of circle centers.  
    pix:
        box size

    Returns
    ----------
    points : float array
        points after centering
        '''
    for i in range(len(points)):
            trackpoint = points[i]
            for j in range(3):#Center 3 times in case the dot has moved partially out of the box during the step.
                
                #draw small boxes
                bf = frame[trackpoint[1]-pix:trackpoint[1]+pix]
                bf = bf[:, trackpoint[0]-pix:trackpoint[0]+pix]
        
                #find center of brightness
                com = ndimage.measurements.center_of_mass(bf.astype('f'))
        
                #find current centers of boxes
                t2 = ones_like(bf, dtype = 'f')
                ce = array(ndimage.measurements.center_of_mass(t2.astype(float)))
                
        
                #find center of mass difference from center of box
                movx = ce[0] - com[0]#pix - com[0]
                movy = ce[1] - com[1]#pix - com[1]
                
                if math.isnan(movx):
                    movx = 0
                if math.isnan(movy):
                    movy = 0
        
                #move the points
                points[i,0] = trackpoint[0] - movy
                points[i,1] = trackpoint[1] - movx
    return points


def find_track(f, frame, output_dir, pix = 2, **kwargs):
    '''
    Finds maxima on a convolution frame.  Draws box around tracked points.  Not currently used with Hoguh transform method.
    
  Parameters
        -----------------
       f: float array
            frame from movie which has been convoluted with desired image kernel
       frame: float array
            original movie frame
       output_dir: string
            string with directory save directory
            
    
        Returns
        ---------------
        points: float array
            list of tracked points in frame
    
    '''
    
    if 'center' in kwargs:
        center = kwargs['center']
    else:
        center = True
    center = True
    if 'order' in kwargs:
        order = kwargs['order']
        
        if order:
            fs = kwargs['fs'] #first set of points in order
        
    else:
        order = False
    
    nn = kwargs['nn']
    
    frame_mean = mean(f.flatten())
    
    #cine.asimage(f).save(output_dir + 'convolved.png')

    #filter for maxima in 60x60 box
    data_max = filters.maximum_filter(f, size = (60,60))
    maxima = (f == data_max)
    
    data_min = filters.minimum_filter(f,size = (60, 60))

    
    dmax = max((data_max-data_min).flatten())
    dmin = min((data_max-data_min).flatten())
    
    minmax = (dmax-dmin)
    
    #find maxima where maxima is 0.45 the maximum data min-max difference
    start_per = .45
    diff = ((data_max - data_min) >= dmin + start_per*minmax)
    diff_num = data_max - data_min
    

    maxima[diff == 0] = 0

    
    
    labeled, num_object = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    
    x,y = [], []
    i = 0
    for dy, dx in slices:
       
        rad = sqrt((dx.stop-dx.start)**2 +(dy.stop-dy.start)**2)
 
        #radius between 25 and 0.2 for maxima
        if rad < 25 and rad >.2:

            x_center = (dx.start+dx.stop)/2.
            y_center = (dy.start + dy.stop)/2.

            if abs(x_center) > 10 and abs(y_center)<590:
                x.append(x_center)
                y.append(y_center)
        i = i + 1

  

    if nn > 0 :
        ntimes = 0
        while len(x)!= nn and ntimes < 100:
            adj = .01
          
            while len(x)> nn and start_per < 1:
                
                
                
                start_per = start_per + adj/(ntimes+1)

                diff = ((data_max - data_min) >= dmin + start_per*minmax)
                diff_num = data_max - data_min
            
        
                maxima[diff == 0] = 0
        
            
            
                labeled, num_object = ndimage.label(maxima)
                slices = ndimage.find_objects(labeled)
            
                x,y = [], []
           
                for dy, dx in slices:
               
                    rad = sqrt((dx.stop-dx.start)**2 +(dy.stop-dy.start)**2)
                    #print 'rad', rad
                
                    if rad < 25 and rad >1.0:
        
                        x_center = (dx.start+dx.stop)/2.
                        y_center = (dy.start + dy.stop)/2.
        
                        if abs(x_center) > 10 and abs(y_center)<590:
                            x.append(x_center)
                            y.append(y_center)
                
                
                
            
            while len(x)< nn and start_per >0:
                
                
                start_per = start_per - adj/(ntimes+1)
                diff = ((data_max - data_min) >= dmin + start_per*minmax)
                diff_num = data_max - data_min
            
        
                maxima[diff == 0] = 0
        
            
            
                labeled, num_object = ndimage.label(maxima)
                slices = ndimage.find_objects(labeled)
            
                x,y = [], []
           
                for dy, dx in slices:
               
                    rad = sqrt((dx.stop-dx.start)**2 +(dy.stop-dy.start)**2)
                    #print 'rad', rad
                
                    if rad < 25 and rad >1.0:
        
                        x_center = (dx.start+dx.stop)/2.
                        y_center = (dy.start + dy.stop)/2.
        
                        if abs(x_center) > 10 and abs(y_center)<590:
                            x.append(x_center)
                            y.append(y_center)
        
            ntimes = ntimes +1


    #get everything to be in the same order all the time.
    x = array(x)
    y = array(y)

    
    points = array([x,y]).T
    if center:
        for i in range(len(points)):
            trackpoint = points[i]
            for j in range(4):
                
                bf = frame[trackpoint[1]-pix:trackpoint[1]+pix]
                bf = bf[:, trackpoint[0]-pix:trackpoint[0]+pix]
        
                com = ndimage.measurements.center_of_mass(bf.astype('f'))
        
                t2 = ones_like(bf, dtype = 'f')
                ce = array(ndimage.measurements.center_of_mass(t2.astype(float)))
                
        
                
                movx = ce[0] - com[0]#pix - com[0]
                movy = ce[1] - com[1]#pix - com[1]
                
                if math.isnan(movx):
                    movx = 0
                if math.isnan(movy):
                    movy =0
        
        
                points[i,0] = trackpoint[0] - movy
                points[i,1] = trackpoint[1] - movx
    if order:
       
        dist, indexes = do_kdtree(fs, points)
       
        points = points[indexes]
        
    return points

def do_kdtree(fs,points, k=1):
  
    mytree = spatial.cKDTree(points)
   
    dist, indexes = mytree.query(fs, k=k)

    
    return dist, indexes


def track_points(f, points, pix, f_num, l_num, output_dir):
    num_points = len(points)
    points = points.astype(float)
    gamma = 2.2

    frames = []
    frames_last = []
    
    for i in range(num_points):
        
        try:
            trackpoint = points[i].astype('f')
           
            
            bf = f[trackpoint[1]-pix:trackpoint[1]+pix]
            bf = bf[:, trackpoint[0]-pix:trackpoint[0]+pix]
            
            sh_bf = array([2*pix, 2*pix])
            
            m_bf = mean(bf)
            std_bf = std(bf)
        
            if f_num ==0:
                if array_equal(shape(bf), sh_bf):
                    frames.append((bf**(1./gamma) * 255).astype('u1'))
                if i == (num_points-1):
                    cine.asimage(hstack(frames)).save(output_dir + 'initial_points' + '.png')
        
        
            for j in range(4):
                bf = f[trackpoint[1]-pix:trackpoint[1]+pix]
                bf = bf[:, trackpoint[0]-pix:trackpoint[0]+pix]
        
                com = ndimage.measurements.center_of_mass(bf.astype('f'))
            
                
            
                t2 = ones_like(bf, dtype = 'f')
                ce = array(ndimage.measurements.center_of_mass(t2.astype(float)))
                            
                
                
                movx = float(ce[0]) - float(com[0])#pix - com[0]
                movy = float(ce[1]) - float(com[1])#pix - com[1]
        
                if math.isnan(movx):
                    movx = 0
                if math.isnan(movy):
                    movy =0
                    

                
                points[i,0] = float(trackpoint[0]) - float(movy)
                points[i,1] = float(trackpoint[1]) - float(movx)

            if f_num ==(l_num-1):
                if array_equal(shape(bf), sh_bf):
                    frames_last.append((bf**(1./gamma) * 255).astype('u1'))
                if i == (num_points-1):
                    cine.asimage(hstack(frames_last)).save(output_dir + 'final_points' + '.png')
                
           
        except RuntimeError:
            points[i] = points[i]
            print 'runtime error'
    
    return points
    
   
mutable_object = {}
def on_key(event):
    N=[event.ydata, event.xdata]
    mutable_object['key'] = N
    plt.close()
    #print N

def find_files_by_extension(root_dir, ext, tot=False):
    filenames  = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(ext):
                if tot == False:
                    filenames.append(file)
                else :
                    filenames.append(root+'/' + file)
            #if file.endswith(ext):
               # filenames.append(root + '/' + file)
    return filenames
                


def find_files_by_name(root_dir, name, tot):
    '''
    Finds files in a root directory (and all directories downstream) by name.  Could be accomplished with 'glob' but is in old code, so keeping it for now.
    
  Parameters
        -----------------
    root_dir: string
        Root directory to look for files in
    name: string
        File name you want to find
    tot: boolean
        If true, returns the full file name. If false, just the root directory
        
    Returns
        -----------------
    filenames: string array
        Array of filenames
    '''
    filenames  = []
    for root, dirs, files in os.walk(root_dir):
        
        if name in files:
            if tot:
                filenames.append(root + '/' + name)
            else :
                filenames.append(root)
                
    return filenames


def isplit(iterable,splitters):
    return [list(g) for k,g in itertools.groupby(iterable,lambda x:x in splitters) if not k]


def color_by_speed(fn, root_dir):
    '''
    Colors gyroscopes by their spinning speed
    
  Parameters
        -----------------
       fn: filename of tracked gyroscope data
       root_dir: directory where files are saved

    
    '''
    
    dat = load_pickled_data(fn)
    speeds = load_pickled_data(root_dir+'speed.pickle')
    
    
    t_dat = (dat.T)[0] 
    num_gyros =  len(t_dat[where(t_dat==0)])

    
    x_dat = (dat.T)[1]
    y_dat = (dat.T)[2]
    
    
    num_time_steps = len(x_dat)/num_gyros
    fig = plt.figure()
    img =  mpimg.imread(root_dir + 'original.png')
    imgplot = plt.imshow(img, cmap= cm.Greys_r)
    
    labels = ['{0}'.format(k) for k in range(num_gyros)]

    x_gy = []
    y_gy = []
    rad = []
    patch = []
    #print len(num_gyros)
    
    for j in range(len(speeds)):
        index = int(speeds[j,0])

        x_gy.append((x_dat[index]))
        y_gy.append((y_dat[index]))
        rad.append(30)
        
        circ = Circle((x_dat[index] ,y_dat[index]), 30)
        patch.append(circ)
    speeds = array(speeds).T[1] 
    
    mean_speeds = mean(abs(speeds))
    std_speeds = std(abs(speeds))
    
    disps =  (abs(speeds)-mean_speeds)/(mean_speeds)
    
    min_lim = mean_speeds - std_speeds
    max_lim = mean_speeds + std_speeds
   
    p_ax = P.axes()
    p = PatchCollection(patch, cmap = 'bwr', alpha = 0.5)
    p.set_array(P.array(abs(speeds)))
    p.set_clim([min_lim, max_lim])
    p_ax.add_collection(p)
    plt.colorbar(p)
   
    p_ax.axes.get_xaxis().set_ticks([])
    p_ax.axes.get_yaxis().set_ticks([])
    
    plt.savefig(root_dir + 'color_by_speed_new.png')
    dump_pickled_data(root_dir, 'disps.pickle', disps)
   
   
def moving_average(values, window):
    weigths = repeat(1.0, window)/window
    smas = convolve(values, weigths, 'valid')
    return smas
    
def check_time(time_s1):
    '''
    Checks to see if it has been long enough since file creation to do tracking.
    
  Parameters
        -----------------
       time_s1: string
            
    
        Returns
        ---------------
        ok_to_continue: boolean
            tells you if it has been long enough since file creation to continue
             
    
    
    '''
    c_time = time.ctime()
    
    #compare years first
    c_time = c_time.split(' ')
    time_s1 = time_s1.split(' ')
    
    c_time_stamp = c_time[3]
    time_s1_stamp = time_s1[3]
    
    c_time[3] = 0
    time_s1 [3] = 0
    
    if c_time != time_s1:
        ok_to_continue = True
    else:
        split_c = c_time_stamp.split(':')
        split_t = time_s1_stamp.split(':')
        
        print split_c
        print split_t
        
        #compare hour
        if split_c[0] != split_t[0]:
            ok_to_continue = True
        else:
            min_c = int(split_c[1])
            min_t = int(split_t[1])
            

            
            time_between = abs(min_c - min_t) #time since creation of file in minutes
  
            if time_between < 3:
                ok_to_continue = False
            else:
                ok_to_continue = True
    return ok_to_continue

def find_nearest(array,value):
    '''
    Finds the index which is nearest to a value in an array
    
  Parameters
        -----------------
       array: an array of floats
       value: Value you're searching for
            
    
        Returns
        ---------------
        idx: integer
            index of entry in array nearest to value
             
    
    
    '''
    idx = (abs(array-value)).argmin()
    return idx



def fft_on_data(dat, output_dir):
    
    '''
    Performs a fourier transform on motor tracked data. colors gyroscopes by speed in an image if maxima of frequencies are over 100
    
  Parameters
        -----------------
       dat: position vs time for each gyroscope in each frame
       output_dir: string specifying output directory
   
    '''
    
    t_dat = (dat.T)[0]

    min_t = min(t_dat)

    num_gyros =  len(t_dat[where(t_dat==min_t)])

    x_dat = (dat.T)[1]
    y_dat = (dat.T)[2]
    
    partitions = 1

    num_time_steps = len(x_dat)/(num_gyros)
    ind = array([k*num_gyros for k in range(num_time_steps)])
    t_f = fft.fftfreq(len(ind), 1)


    tot_power = zeros_like(t_f)
    m_f = []
    fft_list = []
    coords = []

    #get the data ready
    output_dir_1 = output_dir+ 'fourier/'
    output_dir_old = output_dir
    if not os.path.exists(output_dir_1):os.mkdir(output_dir_1)
    copy_dir = output_dir
    
    for u in range(1): #data can be partitioned, but currently I am just putting everything in one partition.
    
        num_in_partition = floor(num_time_steps/partitions)
    
        for j in range(num_gyros):
            pp = u*num_time_steps
            output_dir = os.path.join(output_dir_1, 'gy_%d/' % j)
            if not os.path.exists(output_dir):os.mkdir(output_dir)
        
            ind = array([k*num_gyros+j for k in range(num_time_steps)])
            wind = hanning(len(ind))
        
            x_gy_full = shift(x_dat[ind])
            y_gy_full = shift(y_dat[ind])
            t_gy_full = t_dat[ind]
            
            
            x_gy_full = x_gy_full[:num_in_partition*partitions]
            y_gy_full = y_gy_full[:num_in_partition*partitions]
            t_gy_full = t_gy_full[:num_in_partition*partitions]
            for u in range(partitions):
            
                x_gy = x_gy_full[u*num_in_partition:(u+1)*num_in_partition]
                y_gy = y_gy_full[u*num_in_partition:(u+1)*num_in_partition]
                t_gy = t_gy_full[u*num_in_partition:(u+1)*num_in_partition]
            
    
                wind = hanning(len(t_gy))

                coords.append([mean(x_dat[ind]), mean(y_dat[ind])])
            
                fft_cylindrical_a = fft.fft(wind*array(x_gy + 1j*y_gy))
                fft_cylindrical_negative = fft.fft(wind*array(x_gy - 1j*y_gy))
                fft_cylindrical = abs(fft_cylindrical_a)**2
                fft_cylindrical_n = abs(fft_cylindrical_negative)**2
                fft_x = fft.fft(wind*array(x_gy))
                fft_y = fft.fft(wind*array(y_gy))
                fft_freq = fft.fftfreq(len(t_gy), t_gy[1]-t_gy[0])
            
            
                dump_pickled_data(output_dir, 'x_gy_%01d' % u, x_dat[ind])
                dump_pickled_data(output_dir, 'y_gy_%01d' %u, y_dat[ind])
                dump_pickled_data(output_dir, 'fft_x_%01d' %u , array([fft_freq, fft_x]).T)
                dump_pickled_data(output_dir, 'fft_y_%01d' %u , array([fft_freq, fft_y]).T)
                dump_pickled_data(output_dir, 'fft_complex_%01d_positive' %u , array([fft_freq, fft_cylindrical_a]).T)
                dump_pickled_data(output_dir, 'fft_complex_%01d_negative' %u , array([fft_freq, fft_cylindrical_negative]).T)
                dump_pickled_data(output_dir, 'coords_%01d.pickle' %u, coords[j])
                fft_list.append(fft_cylindrical_a)
    

                
                max_xf = max(fft_cylindrical)
            
                maximum_f = fft_freq[where(fft_cylindrical == max_xf)[0][0]]
                
                if True:#j ==0:
                    fig = plt.figure()
                    ax = fig.add_subplot(1,1,1)
                    plt.plot(array(t_gy), x_gy)
                    plt.plot(array(t_gy), y_gy)
                
                    plt.savefig(output_dir + 'data_xy_%01d.png' %u)
                    print output_dir + 'data_xy_%01d.png' %u
                    
                            
                    lab = ['max of x + iy fft = %0.3f ' % maximum_f, '', '']
                    fig3 = plt.figure()
                    ax = fig3.add_subplot(1,1,1)
                    plt.plot(fft_freq[1:], abs(fft_cylindrical[1:]), label = lab[0])
                    plt.plot(fft_freq[1:], abs(fft_cylindrical_n[1:]), label = lab[0])
                    plt.xlabel('Freq (Hz)', fontsize = 11)
                    plt.ylim(0, max_xf*1.2)
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(handles, labels, fancybox = True, fontsize = 11, loc = 1)
                    #plt.yscale('log')
                    plt.xlim(0,3.5)
                    plt.savefig(output_dir + 'fft_cyl_%01d.png' %u)
                
                    plt.close()

        
                
                if max(fft_freq > 100) :
                    if abs(maximum_f) < 10 :
                
                        selected_f = fft_freq[fft_freq < -100]
                        selected = fft_cylindrical[fft_freq < -100 ]
                    
                
                        max_xf = max(selected)
                        maximum_f = selected_f[where(selected == max_xf)[0][0]]
                
    
                    if abs(maximum_f)>150 and abs(maximum_f)<3200:
                        m_f = list(m_f)
                        m_f.append([j, maximum_f])


                
                plt.close()
            
                if j == 0 :
                    tot_power = zeros_like(fft_cylindrical)
                else:
                   
                    tot_power += fft_cylindrical
       
        if max(fft_freq)>100:
       
            if len(m_f)>1:
                dump_pickled_data(copy_dir, '/speed', array(m_f))
                m_f = array(m_f).T[1]
                fig55 = plt.figure()
                ax = fig55.add_subplot(111)
                n, bins, patches = ax.hist(abs(array(m_f)), 100, normed=False, facecolor='green', alpha=0.75)
                plt.savefig(copy_dir + '/speed.png')
                m_f = array(m_f)
                plt.close()
                
                print output_dir_old + 'com_data.pickle'
                color_by_speed(output_dir_old + 'com_data.pickle', output_dir_old)

def dump_text_data(output_dir, filename, data):
    con = open(output_dir + '/' + filename + '.csv', "wb")
    con_len = len(data)
    data = array(data)
    print data[0]
    print data[con_len-1]

    for i in range(con_len):
        for j in range(len(data[0])):
            #print i, j
            #print data[i,j]
            con.write(str(data[i,j]) + ' ,') 
        con.write('\n')
    con.close()

def dump_pickled_data(output_dir, filename, data):
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    of = open(output_dir + '/'+filename + '.pickle', 'wb')
    pickle.dump(data, of, pickle.HIGHEST_PROTOCOL)
    of.close()
       
   
def shift(yvals):
    #centers values
    y_mean = mean(yvals)
    yvals = yvals - y_mean
    return yvals

def pick_point(fn, ff, c):
    #Pick point on click 
    minval, maxval = 15, 60#60, 2000
    frame = (clip(c[ff].astype('f'), minval, maxval)-minval)/(maxval-minval)
    #frame = c[0].astype('f')/2**c.real_bpp
    m_bf = mean(frame)
    std_bf = std(frame)
        
    minval = m_bf -10.*std_bf
    maxval = m_bf + 500.*std_bf

    f = (clip(c[ff].astype('f'), minval, maxval)-minval)/(maxval-minval)
    cine.asimage(f).save(fn + '.png')

    fig = plt.figure()
    img =  mpimg.imread(fn + '.png')
    imgplot = plt.imshow(img, cmap= cm.Greys_r)


    fig.canvas.mpl_connect('button_press_event', on_key)
    plt.show()
    N = mutable_object['key']
    #print array(N)[0] 
    
    return array(N)
