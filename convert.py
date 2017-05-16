import cv2

vidcap = cv2.VideoCapture('/Users/lisa/Dropbox/Research/disordered_experiment/2017_05_14/sweep.MOV')

success, image = vidcap.read()

count = 0;
while success:
    success, image = vidcap.read()

    cv2.imwrite("/Users/lisa/Dropbox/Research/disordered_experiment/2017_05_14/sweep/frames/frame_%05d.png" % count,
                image)  # save frame as JPEG file
    count += 1
