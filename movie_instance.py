import cine
import cv2
import numpy as np
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt
import math
import PIL.Image as Image
import PIL.ImageOps as ImageOps
import matplotlib.cm as cm


class GyroMovieInstance:
    def __init__(self, input_file, frame_rate=60):
        # first determind the file type
        self.file_type = input_file.split('.')[-1]

        if self.file_type == 'cine':
            self.cine = True
            self.data = cine.Cine(input_file)
        else:
            self.cine = False

        self.num_frames = len(self.data)
        self._mean_value = 0
        self.min_radius = 17
        self.max_radius = 22
        self._min_value = 0.05
        self._max_value = 0.7
        self._pix = 6

        self.current_frame = []
        self.frame_current_points = []
        self.circles = []
        self.current_time = 0
        self.frame_rate = frame_rate

        self._adjust_min_max_val()
        self._set_dummy_frame()


    def _adjust_min_max_val(self):
        max = np.max(self.data[0].astype('float').flatten())
        self._min_value = self._min_value * max
        self._max_value = self._max_value * max



    def _set_dummy_frame(self):
        t2 = np.ones((2 * self._pix, 2 * self._pix), dtype='f')
        self.dummy = np.array(ndimage.measurements.center_of_mass(t2.astype(float)))

    def set_min_max_val(self, min_value, max_value):
        self._min_value = min_value
        self._max_value = max_value
        self._adjust_min_max_val()

    def set_tracking_size(self, pix):
        self._pix = pix
        self._set_dummy_frame()

    def extract_frame_data(self, frame_num):
        self.current_frame = self.data[frame_num].astype('float')
        self.get_time(frame_num)


    def get_time(self, frame_num):
        if self.cine:
            self.current_time = self.data.get_time(frame_num)

        else:
            print('...frame rate set to %02d...' % frame_rate)
            self.current_time = 1. / frame_rate * frame_num

    def adjust_frame(self):
        self.current_frame = np.clip(self.current_frame, self._min_value, self._max_value) - self._min_value
        self.current_frame = self.current_frame / (self._max_value - self._min_value)
        self._mean_value = np.mean(self.current_frame)

    def find_points_hough(self):
        img = np.array(self.current_frame * 255, dtype=np.uint8)

        # apply blur so you don't find lots of fake circles
        img = cv2.GaussianBlur(img, (3, 3), 2, 2)

        circles = cv2.HoughCircles(img, cv2.cv.CV_HOUGH_GRADIENT, 1, 20,
                                   param1=48, param2=18, minRadius=self.min_radius, maxRadius=self.max_radius)

        circles = np.uint16(np.around(circles))
        self.circles = circles[0]

        self.frame_current_points = np.array([self.circles[:, 0], self.circles[:, 1]], dtype=float).T

    def center_on_bright(self, num_times=3):
        new_points = []

        for pt in self.frame_current_points:

            w, h = np.shape(self.current_frame)
            if ((pt[0] > 1.5*self._pix) and (pt[1] > 1.5*self._pix) and (pt[0] < w - 1.5*self._pix) and (pt[1] < h - 1.5*self._pix)):
                for j in xrange(num_times):
                    # Center num_times in case the dot has moved partially out of the box during the step.
                    # draw small boxes
                    bf = self.current_frame[pt[1] - self._pix:pt[1] + self._pix]
                    bf = bf[:, pt[0] - self._pix:pt[0] + self._pix]
                    bf_comp = bf.copy()
                    # let's clip this area to maximize the bright spot
                    bf = bf.astype('f')

                    bf_min = 0.97 * np.min(bf.flatten())
                    bf_max = 1. * np.max(bf.flatten())
                    bf = np.clip(bf, bf_min, bf_max) - bf_min
                    bf = bf / (bf_max - bf_min)


                    # find center of brightness
                    com = ndimage.measurements.center_of_mass(bf)

                    # if j == num_times -1:
                    #     fig = plt.figure()
                    #     plt.imshow(bf)
                    #     plt.show()

                    # find center of mass difference from center of box
                    movx = self.dummy[1] - com[1]  # pix - com[0]
                    movy = self.dummy[0] - com[0]  # pix - com[1]

                    if math.isnan(movx):
                        movx = 0
                    if math.isnan(movy):
                        movy = 0

                    # move the points
                    pt[0] = pt[0] - movx
                    pt[1] = pt[1] - movy



                if np.mean(bf_comp)<5*self._mean_value:
                    new_points.append(pt)

        new_points = np.array(new_points, dtype=float)
        ind = np.argsort(new_points[:, 0])
        new_points = new_points[ind]
        ind = np.argsort(new_points[:, 1])
        new_points = new_points[ind]

        self.frame_current_points = np.array(new_points, dtype=float)

    def save_frame(self, name='frame'):
        fig = plt.figure()
        ax = plt.axes([0, 0, 1, 1])
        img = cine.asimage(self.current_frame)
        plt.imshow(img, cmap=cm.Greys_r)
        plt.savefig(name + '.png')
        plt.close()

    def save_frame_with_boxes(self, name='frame'):
        fig = plt.figure()
        ax = plt.axes([0, 0, 1, 1])
        img = np.array(self.current_frame)

        for pt in self.frame_current_points:
            img[pt[1] - self._pix: pt[1] + self._pix, pt[0] - self._pix: pt[0] + self._pix] = np.array(
                ImageOps.invert(Image.fromarray(np.uint8(
                    img[pt[1] - self._pix: pt[1] + self._pix, pt[0] - self._pix: pt[0] + self._pix]))))

        img = cine.asimage(img)
        plt.imshow(img, cmap=cm.Greys_r)
        plt.savefig(name + '.png')
        plt.close()

    def find_point_convolve(self, img_ker):
        fr = ndimage.convolve(self.current_frame, img_ker, mode='reflect', cval=0.0)
        minval = 0.1 * max(fr.flatten())
        maxval = 1 * max(fr.flatten())
        fr = (np.clip(fr, minval, maxval) - minval) / (maxval - minval)

        fig = plt.figure()
        plt.imshow(fr)
        plt.show()
