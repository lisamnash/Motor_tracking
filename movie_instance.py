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
    def __init__(self, input_file):
        # first determind the file type
        self.file_type = input_file.split('.')[-1]

        if self.file_type == 'cine':
            self.cine = True
            self.data = cine.Cine(input_file)

        self.num_frames = len(self.data)

        self.min_radius = 17
        self.max_radius = 22
        self._min_value = 0.05
        self._max_value = 0.7
        self._pix = 6

        self.current_frame = []
        self.frame_current_points = []
        self.circles = []

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

    def adjust_frame(self):
        self.current_frame = np.clip(self.current_frame, self._min_value, self._max_value) - self._min_value
        self.current_frame = self.current_frame / (self._max_value - self._min_value)

    def find_points_hough(self):
        img = np.array(self.current_frame * 255, dtype=np.uint8)

        # apply blur so you don't find lots of fake circles
        img = cv2.GaussianBlur(img, (3, 3), 2, 2)

        circles = cv2.HoughCircles(img, cv2.cv.CV_HOUGH_GRADIENT, 1, 20,
                                   param1=48, param2=18, minRadius=self.min_radius, maxRadius=self.max_radius)

        circles = np.uint16(np.around(circles))
        self.circles = circles[0]

        self.frame_current_points = np.array([self.circles[:, 0], self.circles[:, 1]]).T

    def center_on_bright(self, num_times=3):
        new_points = []
        for pt in self.frame_current_points:
            for j in xrange(num_times):
                # Center num_times in case the dot has moved partially out of the box during the step.
                # draw small boxes

                bf = self.current_frame[pt[1] - self._pix:pt[1] + self._pix]
                bf = bf[:, pt[0] - self._pix:pt[0] + self._pix]

                # let's clip this area to maximize the bright spot
                bf = bf.astype('f')
                bf_min = 0.9 * np.min(bf.flatten())
                bf_max = 0.95 * np.max(bf.flatten())
                bf = np.clip(bf, bf_min, bf_max) - bf_min
                bf = bf / (bf_max - bf_min)

                # find center of brightness
                com = ndimage.measurements.center_of_mass(bf)

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

            new_points.append(pt)

        self.frame_current_points = np.array(new_points)

    def save_frame(self, pt_save=True, name='frame'):
        fig = plt.figure(figsize=(3, 3))
        ax = plt.axes([0, 0, 1, 1])
        img = cine.asimage(self.current_frame)
        plt.imshow(img, cmap=cm.Greys_r)
        plt.savefig(name + '.png')
        plt.close()

    def save_frame_with_boxes(self, pt_save=True, name='frame'):
        fig = plt.figure(figsize=(3, 3))
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
