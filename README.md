# 3d_avatar_with_facetracking
## Facetracking is completely based on mediapipe(Python)
this is complete and working code, but I suspect that my webcam cannot handle more than 30 fps, so in my case
the fps mean was around 24, I cannot guarantee that fps will rise that significantly with a better webcam, but I
think that it is worth trying. The avatar is not supposed to be changed in shape, if you wanna load your own shape,
you can simply create a mesh in blender(leaving only Triangulate faces option while exporting) and load it but do not 
make too complicated stuff if you do not want you fpf to be 0). If you really decide on changing the shape just 
use my distortion function to change the perspective of an image using cv2 for decorations.

# Features
* facial orientation and position tracking
* based on a simple 3d engine in pygame