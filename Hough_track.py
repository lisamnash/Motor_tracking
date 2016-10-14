import argparse
import sys
import cine
from numpy import *
import matplotlib.pyplot as plt
import cPickle as pickle
import scipy.ndimage as image
import motor_track_functions as mtf
import PIL.Image as Image
import matplotlib.image as mpimg
import scipy.ndimage as ndimage
import PIL.ImageOps as ImageOps
import numpy as np
import matplotlib.cm as cm
import time
import os
import cv2



if __name__ == '__main__':
    '''example usage python Hough_track.py -i /Users/lisa/Dropbox/research/lisa/Motor_tracking_code/test_cine/2_Cam_7326_Cine3.cine -min_radius 25 -max_radius 33 -o /Users/lisa/Dropbox/research/lisa/Motor_tracking_code -min_val 0.02 -max_val 0.4 -pix 4'''
    parser = argparse.ArgumentParser(description = 'Track points from a video')
    
    parser.add_argument('-i', dest='input_file', type=str, help='input cine file. required', required= True)
    parser.add_argument('-o', dest='output_directory', type=str, help='output directory.', default = './')
    parser.add_argument('-k', dest = 'img_ker', type=str, help='image kernel', default = "./new_imker.png")
    parser.add_argument('-ff', dest = 'ff', type=int, help='first frame to track', default = 0)
    parser.add_argument('-lf', dest = 'lf', type=int, help='last frame to track')
    parser.add_argument('-append_fn', type = str, help = 'name to append to output directory for descriptive purposes' )
    parser.add_argument('-min_val', type=float, help='minimum brightness', default = 0.05)
    parser.add_argument('-max_val', type=float, help='maximum brightness', default = 0.3)
    parser.add_argument('-monitor', type = bool, help = 'Monitor first frame and tracking box size? (boolean)', default = False)
    parser.add_argument('-pix', type = float, help = "Length of tracking box", default = 4)
    parser.add_argument('-cf', dest = 'cf', nargs = '+', type = int, help = 'Frames to perform convolution. Use this if the point gets lost in a frame.', default = [0])

    parser.add_argument('-fft_only', type = bool, help = 'Just perform FFT on already tracked data?', default = False)
    parser.add_argument('-min_radius', type = int, help ='Minimum radius for Hough transform', default = 15)
    parser.add_argument('-max_radius', type = int, help = 'Maximum radius for Hough transform', default = 33)
    parser.add_argument('-tracked_image', type = bool, help= 'If True, plots all the frames with tracked circles.', default = False)
    parser.add_argument('-save_during', type = bool, help = 'If True, saves the points at each time step in a folder.  Does not save com_data file', default = False)
    parser.add_argument('-save_com', type = bool, help = 'If True, saves com_data file', default = True)
    parser.add_argument('-skip_Hough', dest = 'skip_Hough', type = bool, help = 'If True, skips Hough transform on all but the first frame', default = False)
    
    
    args = parser.parse_args()

    fns = []
    if args.input_file:
        fn = args.input_file
        
        spl = fn.split('.')[-1]
        print 'spl', spl
        if spl == 'cine':
            fns = [fn]
        else:
            #find all cines in directory
            fns = mtf.find_files_by_extension(fn, '.cine', tot= True)
            print fns
            
    else:
        print 'No input file selected.  Exiting'
        sys.exit()
    
    
    plot_tr = args.tracked_image
    max_val = args.max_val
    min_val = args.min_val
    min_rad = args.min_radius
    max_rad = args.max_radius
    save_com = args.save_com
    save_during = args.save_during
    sH = args.skip_Hough

    
    con_all = True
    
    for ii in range(len(fns)):
        c = cine.Cine(fns[ii])
        fn = fns[ii]
        ff = args.ff
        if args.lf:
            lf = args.lf
        else:
            tot_frames = len(c)
            lf = tot_frames

        output_dir = args.output_directory+'/'
        output = fn.split('/')[-1]
        output = output.split('.')[0]
        
        
        con_f = args.cf
       
        
        if con_f[0] != args.ff:
            con_f.insert(0, args.ff)

        
        if con_all:
            for kk in range(lf-ff):
                con_f.insert(-1,ff+kk)
            con_f = list(set(con_f))

      
        if args.append_fn:
            output = output + '_' + args.append_fn
        output = output + '/'
        print output_dir + output
        if not os.path.exists(output_dir): os.mkdir(output_dir)
        if not os.path.exists(output_dir + output): os.mkdir(output_dir + output)
        
        output_dir = output_dir + output
        output_dir_images = output_dir + '/images'
        
        if not os.path.exists(output_dir_images):os.mkdir(output_dir_images)

        if save_during:
            sd_dir = output_dir + '/steps'
            if not os.path.exists(sd_dir):os.mkdir(sd_dir)
        
        if args.monitor :
            trackpoint = mtf.pick_point(fn, ff, c)
            ent_frame = raw_input('Continue (y) or choose a different first frame (any other keyed entry) : ')
        
            while ent_frame != 'y':
                ff= input('Enter first frame number : ')
                lf = input('Enter last frame number : ')
                lf = int(lf)
                ff = int(ff)
                
                if lf == -1:
                    lf = len(c)
        
            
                trackpoint = mtf.pick_point(fn, ff, c)
            
            
                ent_frame = raw_input('Continue (y) or choose a different first frame (any other keyed entry) : ' )
            
       
        val = 1 * max(c[ff].astype(float).flatten())
        minval = min_val*val
        maxval = max_val*val
        
        com_data = []
        tot_frames = lf
        bad_frames = []
        for i in range(tot_frames-ff):

            if not sH or i ==0 :
                print 'not sH'
                try:
            
                    frame = (clip(c[ff+i].astype('f'), minval, maxval)-minval)/(maxval-minval)
                except:
                    frame = c[ff+i-1]
               

                
                if i == 0:
                     original = cine.asimage(frame)
                     cine.asimage(frame).save(output_dir + 'original.png')

                     print 'The output directory is', output_dir
                     od = False
                     fs =[-1]
                     nn = -1
                else:
                     od = True
                     nn = len(fs)
                     
         
                #find the circles
                points, circles, frame, flag = mtf.find_circles(frame, fs, order = od, min_rad = min_rad, max_rad =max_rad) 
                
                ent_pix = '-3.14159'
               
                         
         
                
                if ff+i == ff or plot_tr:
                   
                    pix = args.pix
                    fig = plt.figure(figsize = (8,8))
                    ax = plt.axes([0,0,1,1])
                    img =  cine.asimage(frame)#Image.open(output_dir + 'original.png')
                    img = array(img)
                    img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
                    for j in range(len(points)):    
                        img[points[j,1]-pix: points[j,1]+pix, points[j,0]-pix: points[j,0]+pix] = array(ImageOps.invert(Image.fromarray(np.uint8(img[points[j,1]-pix: points[j,1]+pix, points[j,0]-pix: points[j,0]+pix]))))
                    plt.xticks([])
                    plt.yticks([])
                     
                    for ij in circles[0,:]:
                         # draw the outer circle
                    
                        cv2.circle(img,(ij[0],ij[1]),ij[2],(0,255,0),2)
                     
                    plt.imshow(img)
                    plt.savefig(output_dir_images + '/tracked_%05d.png'%(i+ff))
                    #plt.show()
                    plt.close()
                     
       
                fs  = points

                      

                #you might want to change how this works.  I use flag to append the data from this step or not, but you could use it to decide whether or not to append the data from the previous time step.
                
        
                
            else:
                try:
            
                    frame = (clip(c[ff+i].astype('f'), minval, maxval)-minval)/(maxval-minval)
                except:
                    frame = c[ff+i-1]
               
                points = mtf.track_points(frame.astype('f'), points, pix, i, tot_frames-ff, output_dir).astype(float)
                
            t = array([[c.get_time(ff+i), points[j,0], points[j,1]] for j in range(len(points))])
            
            
            if ff+i == ff or plot_tr:
                   
                pix = args.pix
                fig = plt.figure(figsize = (8,8))
                ax = plt.axes([0,0,1,1])
                img =  cine.asimage(frame)#Image.open(output_dir + 'original.png')
                img = array(img)
                img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
                for j in range(len(points)):    
                    img[points[j,1]-pix: points[j,1]+pix, points[j,0]-pix: points[j,0]+pix] = array(ImageOps.invert(Image.fromarray(np.uint8(img[points[j,1]-pix: points[j,1]+pix, points[j,0]-pix: points[j,0]+pix]))))
                plt.xticks([])
                plt.yticks([])
                
                for ij in circles[0,:]:
                    # draw the outer circle
                    
                    cv2.circle(img,(ij[0],ij[1]),ij[2],(0,255,0),2)
                     
              
                plt.imshow(img)
                plt.savefig(output_dir_images + '/tracked_%05d.png'%(i+ff))
                #plt.show()
                plt.close()
                
            if save_during:
                mtf.dump_pickled_data(sd_dir, 'frame_%05d'%(i+ff), t)

            if flag:
                com_data.append(t)
            else:
                bad_frames.append(ff+i)
                print 'bad frame', ff+i
                pix = args.pix
  
            if i % 100 == 0:
                print i

        #currently the data is output all at once at the end of the code.
        com_data = array(com_data)
        sh1, sh2, sh3 = shape(com_data)
        com_data = com_data.flatten()
        com_data = reshape(com_data, [sh1*sh2,3])
        

        if save_com:  
            mtf.dump_pickled_data(output_dir, 'com_data', com_data)
        mtf.fft_on_data(com_data, output_dir)
        
        