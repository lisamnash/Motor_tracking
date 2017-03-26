# Motor_tracking
Motor tracking code

To use:
Change the input directory to your cine's directory (also change the output directory accordingly).  

Other setting parameters:

-`first_frame` : first frame to track

`last_frame`: last frame to track, or set to -1 to track until the end of the movie

-`pix`: tracking box size

-`min_max_radius`: [min_radius, max_radius] of circle you're looking for

`cf`: which frames to perform Hough transform on (bright dot tracked in tracking box for the rest of the frames]

`min_max_val`: adjusts the frame brightness.

I use `pipeline.py` to run the code, but you will need to comment out the part that makes movies since my movie tracking code isn't up/up to date.
