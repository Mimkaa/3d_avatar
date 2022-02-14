import pygame as pg
import sys
from settings import *
import cv2
from objects import *
import numpy as np
import math
from os import path
from webcam_reading_mediapipe import *
from perspective import distortion
from stripe_management import Running_Stripes
vec2=pg.Vector2
def distance(x1,y1,x2,y2):
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist

vec2=pg.Vector2
class Game:
    def __init__(self):
        pg.init()
        self.screen = pg.display.set_mode((WIDTH, HEIGHT))
        pg.display.set_caption(TITLE)
        self.clock = pg.time.Clock()
        pg.key.set_repeat(500, 100)
        self.load_data()

    def load_data(self):
        self.font=path.join("PixelatedRegular-aLKm.ttf")

    def draw_text(self, text, font_name, size, color, x, y, align="nw"):
        font = pg.font.Font(font_name, size)
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if align == "nw":
            text_rect.topleft = (x, y)
        if align == "ne":
            text_rect.topright = (x, y)
        if align == "sw":
            text_rect.bottomleft = (x, y)
        if align == "se":
            text_rect.bottomright = (x, y)
        if align == "n":
            text_rect.midtop = (x, y)
        if align == "s":
            text_rect.midbottom = (x, y)
        if align == "e":
            text_rect.midright = (x, y)
        if align == "w":
            text_rect.midleft = (x, y)
        if align == "center":
            text_rect.center = (x, y)
        self.screen.blit(text_surface, text_rect)
        return text_rect

    def new(self):
        # initialize all variables and do all the setup for a new game
        self.curr_mesh=Mesh([
            # SOUTH
            [[0.,0.,0.],[0.,1.,0.],[1.,1.,0.]],
            [[0.,0.,0.],[1.,1.,0.],[1.,0.,0.]],
            # EAST
            [[1.,0.,0.,],[1.,1.,0.,],[1.,1.,1.]],
            [[1.,0.,0.,],[1.,1.,1.,],[1.,0.,1.]],
            # NORTH
            [[1.,0.,1.],[1.,1.,1.],[0.,1.,1.]],
            [[1.,0.,1.],[0.,1.,1.],[0.,0.,1.]],
            # WEST
            [[0.,0.,1.],[0.,1.,1.],[0.,1.,0.]],
            [[0.,0.,1.],[0.,1.,0.],[0.,0.,0.]],
            # TOP
            [[0.,1.,0.],[0.,1.,1.],[1.,1.,1.]],
            [[0.,1.,0.],[1.,1.,1.],[1.,1.,0.]],
            # BOTTOM
            [[1.,0.,1.],[0.,0.,1.],[0.,0.,0.]],
            [[1.,0.,1.],[0.,0.,0.],[1.,0.,0.]],
        ])
        for tri in self.curr_mesh.tris:
            for v in tri.vectors:
                v.x-=0.5
                v.y-=0.5

        self.curr_mesh.LoadFromObjectFile('head.obj')

        # gives triangles indices
        for n,t in enumerate(self.curr_mesh.tris):
            t.index=n

        #self.fTheta=-math.pi
        self.fTheta=0
        # projection matrix
        self.near=0.1
        self.far=1000.
        self.fov=90.
        self.aspectratio=self.screen.get_height()/self.screen.get_width()
        self.Fovrad=1/math.tan(math.radians(self.fov*0.5))

        self.matProj=np.array([[self.aspectratio*self.Fovrad,0,0,0],
                               [0,self.Fovrad,0,0],
                               [0,0,self.far/(self.far-self.near),1.],
                               [0,0,(-self.far*self.near)/(self.far-self.near),0]])

        self.camera=ProtoVector(0,0,0)
        self.LookDir=ProtoVector(0,0,1)
        self.Up=ProtoVector(0,1,0)
        self.Yaw=0.

        # for face reading

        self.rot_z=0.
        self.rot_y=0.
        self.rot_y_backup=[]
        self.rot_x=0.


        self.list_face_points=[]
        self.capture_web=cv2.VideoCapture(0)
        self.original_eye_dist=0
        self.original_pos=[]
        self.angles_back_up=[]
        self.zoom_ratio=0
        self.x_ratio=0
        self.y_ratio=0

        self.left_eye_ratios=[]
        self.left_eye_data=[]
        self.left_eye_clip=False
        self.left_eye_cur=0
        self.right_eye_ratios=[]
        self.right_eye_data=[]
        self.right_eye_clip=False
        self.right_eye_cur=0
        self.blinking=True

        # mouth_width_height

        self.m_height=0
        self.m_width=0
        self.m_offset=vec2(0,0)

        #self.pol_history=[]

        # stripes
        self.stripes=Running_Stripes([pg.Surface((150,150))  for i in range(15)],(PURPLE1,PURPLE2,PURPLE3))
        self.s_surfs=[]
        self.s_size=(150,150)

        # for_gradual_colour_changing
        self.color_speed=100
        self.color_dir=[1,1,1]
        self.default_color=[123,129,249]
        self.max_col=251
        self.min_col=123

    def gradual_color_changing(self):
        for i in range(3):
            self.default_color[i]+=int(self.color_speed*self.color_dir[i]*self.dt)
            if self.default_color[i]>=self.max_col:
                self.color_dir[i]*=-1
                self.default_color[i]=self.max_col
            elif self.default_color[i]<=self.min_col:
                self.color_dir[i]*=-1
                self.default_color[i]=self.min_col


    def create_z_rot_mat(self,angle):
        rotation_z=np.array([
            [math.cos(angle),math.sin(angle),0.,0.],
            [-math.sin(angle),math.cos(angle),0.,0.],
            [0.,0.,1.,0.],
            [0.,0.,0.,1.]
        ])
        return rotation_z

    def create_x_rot_mat(self,angle):
        rotation_x=np.array([
            [1.,0.,0.,0.],
            [0.,math.cos(angle*0.5),math.sin(angle*0.5),0.],
            [0.,-math.sin(angle*0.5),math.cos(angle*0.5),0.],
            [0.,0.,0.,1.]
        ])
        return rotation_x

    def create_y_rot_mat(self,angle):
        rotation_y=np.array([
            [math.cos(angle),0,math.sin(angle),0],
            [0,1,0,0],
            [-math.sin(angle),0,math.cos(angle),0],
            [0,0,0,1]
        ])
        return rotation_y

    def MatrixPointAt(self,pos,target,up):
        # new forward vector
        forward=target-pos
        forward=forward.normalize()

        # new up vector
        a=forward*up.dot(forward)
        newUp=up-a
        newUp.normalize()

        # new right vector
        newRight=newUp.cross(forward)

        # Dimensioning and translation Matrix
        mat=np.array([
            [newRight.x,newRight.y,newRight.z,0.],
            [newUp.x,newUp.y,newUp.z,0.],
            [forward.x,forward.y,forward.z,0.],
            [pos.x,pos.y,pos.z,1.]
        ])

        return mat

    def MatInverse(self,mat):
        return np.linalg.inv(mat)


    def Vector_IntersectPlane(self, point_on_plane, normal_on_plane, line_start, line_end):
        normal_on_plane=normal_on_plane.normalize()

        plane_d=-normal_on_plane.dot(point_on_plane)
        ad=line_start.dot(normal_on_plane)
        bd=line_end.dot(normal_on_plane)
        t=(-plane_d-ad)/(bd-ad)
        lineStartEnd=line_end-line_start
        lineToIntesect=lineStartEnd*t
        return line_start+lineToIntesect

    def dist(self,vecc,plane_n,plane_p):
        plane_n.normalize()
        return (plane_n.x*vecc.x+plane_n.y*vecc.y+plane_n.z*vecc.z-plane_n.dot(plane_p))

    def Triange_ClipAgainstPlane(self, point_on_plane, normal_on_plane, in_trii):
            out_tri1=None
            out_tri2=None
            in_tri=in_trii.copy()

            # plane normal in indeed normal
            normal_on_plane=normal_on_plane.normalize()
            # if distance sign is positive , point is inside of the plain
            inside_points=[]
            outside_points=[]
            count_inside=0
            count_outside=0
            # for textures
            inside_points_tex=[]

            # get signed dists of each point in tri to plane
            d0=self.dist(in_tri.vectors[0],normal_on_plane,point_on_plane)
            d1=self.dist(in_tri.vectors[1],normal_on_plane,point_on_plane)
            d2=self.dist(in_tri.vectors[2],normal_on_plane,point_on_plane)

            if d0>=0:
                count_inside+=1
                inside_points.append(in_tri.vectors[0])

            else:
                count_outside+=1
                outside_points.append(in_tri.vectors[0])

            if d1>=0:
                count_inside+=1
                inside_points.append(in_tri.vectors[1])

            else:
                count_outside+=1
                outside_points.append(in_tri.vectors[1])

            if d2>=0:
                count_inside+=1
                inside_points.append(in_tri.vectors[2])

            else:
                count_outside+=1
                outside_points.append(in_tri.vectors[2])


            # classify triangles
            if count_inside==0:
                return 0,[out_tri1,out_tri2]
            if count_inside==3:
                # out_tri1=Triangle(in_tri.vectors)
                # out_tri1.light_dp=in_tri.light_dp
                out_tri1=in_tri.copy()

                return 1,[out_tri1,out_tri2]
            if count_inside==1 and count_outside==2:
                # out_tri1=Triangle(in_tri.vectors)
                # out_tri1.light_dp=in_tri.light_dp
                out_tri1=in_tri.copy()
                # inside point is valid
                out_tri1.vectors[0]=inside_points[0]


                # two other will be where triangle intersect the plain
                out_tri1.vectors[1]=self.Vector_IntersectPlane(point_on_plane,normal_on_plane,inside_points[0],outside_points[0])

                out_tri1.vectors[2]=self.Vector_IntersectPlane(point_on_plane,normal_on_plane,inside_points[0],outside_points[1])

                return  1,[out_tri1,out_tri2]
            if count_inside==2 and count_outside==1:

                # out_tri1=Triangle(in_tri.vectors)
                # out_tri1.light_dp=in_tri.light_dp
                out_tri1=in_tri.copy()

                # out_tri2=Triangle(in_tri.vectors)
                # out_tri2.light_dp=in_tri.light_dp
                out_tri2=in_tri.copy()
                # first triangle 2 inside points and a new point on the place of the intersection
                out_tri1.vectors[0]=inside_points[0]


                out_tri1.vectors[1]=inside_points[1]


                out_tri1.vectors[2]=self.Vector_IntersectPlane(point_on_plane,normal_on_plane,inside_points[0],outside_points[0])


                # and the second one
                out_tri2.vectors[0]=inside_points[1]

                out_tri2.vectors[1]=out_tri1.vectors[2]

                out_tri2.vectors[2]=self.Vector_IntersectPlane(point_on_plane,normal_on_plane,inside_points[1],outside_points[0])

                return 2,[out_tri1,out_tri2]





    def MultiplyMatrixVector(self,vec,mat):
        arr=np.array([vec.x,vec.y,vec.z,vec.w])
        result=arr.dot(mat)
        new=ProtoVector(result[0],result[1],result[2])
        new.w=result[3]
        return new




    def run(self):
        # game loop - set self.playing = False to end the game
        self.playing = True
        while self.playing:
            self.list_face_points=webcam_frames_m(self.capture_web)
            self.list_face_points=[p for p in self.list_face_points]
            self.dt = self.clock.tick(FPS) / 1000
            self.events()
            self.update()
            self.draw()

    def quit(self):
        pg.quit()
        sys.exit()

    def update(self):
        # update portion of the game loop


        self.camera=ProtoVector(0,0,0)
        # gradual color
        self.gradual_color_changing()

        #stripe

        self.stripes.update(self.dt)
        self.s_surfs=self.stripes.render()
        # for n,s in enumerate(self.s_surfs):
        #     self.s_surfs[n]=pg.transform.scale(s,self.s_size)
        # face angles


        if self.list_face_points:

            eyes_dist=distance(self.list_face_points[386][0],self.list_face_points[386][1],self.list_face_points[160][0],self.list_face_points[160][1])

            if self.original_eye_dist==0:
                self.original_eye_dist=eyes_dist
            if not self.original_pos:
                self.original_pos=[self.list_face_points[160][0],self.list_face_points[160][1]]
            self.rot_z=math.atan2(self.list_face_points[386][1]-self.list_face_points[160][1],self.list_face_points[386][0]-self.list_face_points[160][0])

            self.rot_y_backup.append(self.rot_y)
            if len(self.rot_y_backup)>3:
                self.rot_y_backup.pop(0)
            left_eye_brow_length=distance(self.list_face_points[113][0],self.list_face_points[113][1],self.list_face_points[221][0],self.list_face_points[221][1])/eyes_dist
            right_eye_brow_length=distance(self.list_face_points[441][0],self.list_face_points[441][1],self.list_face_points[467][0],self.list_face_points[467][1])/eyes_dist
            eyes_imp=left_eye_brow_length-right_eye_brow_length

            self.rot_y=math.radians(eyes_imp*90)*0.85





            nose_length=distance(self.list_face_points[1][0],self.list_face_points[1][1],self.list_face_points[164][0],self.list_face_points[164][1])/eyes_dist
            #cheek_nose_bottom=distance(self.list_face_points[147][0],self.list_face_points[147][1],self.list_face_points[164][0],self.list_face_points[164][1])/eyes_dist
            self.rot_x=math.radians(math.degrees(-math.atan2(nose_length,0.05)+math.radians(66))*3)
            if self.rot_x>0.6:
                self.rot_x=0.6





            self.zoom_ratio=(eyes_dist/self.original_eye_dist)
            self.x_ratio=(self.original_pos[0]-self.list_face_points[160][0])/self.original_pos[0]
            self.y_ratio=(self.original_pos[1]-self.list_face_points[160][1])/self.original_pos[1]
            self.camera.z+=self.zoom_ratio
            self.camera.x+=self.x_ratio
            self.camera.y+=self.y_ratio

            # eye_ratios
            # self.left_eye_ratios.append(distance(self.list_face_points[386][0],self.list_face_points[386][1],self.list_face_points[374][0],self.list_face_points[374][1])/eyes_dist)
            # if len(self.left_eye_ratios)>3:
            #     self.left_eye_ratios.pop(0)
            #
            # if distance(self.list_face_points[144][0],self.list_face_points[144][1],self.list_face_points[160][0],self.list_face_points[160][1])/eyes_dist<round(sum(self.right_eye_ratios)/3,2):
            #     self.right_eye_clip=True
            # else:
            #     self.right_eye_clip=False
            # self.right_eye_ratios.append(distance(self.list_face_points[144][0],self.list_face_points[144][1],self.list_face_points[160][0],self.list_face_points[160][1])/eyes_dist)
            # if len(self.right_eye_ratios)>3:
            #     self.right_eye_ratios.pop(0)

            dist_ver1=distance(self.list_face_points[386][0],self.list_face_points[386][1],self.list_face_points[374][0],self.list_face_points[374][1])
            dist_ver2=distance(self.list_face_points[380][0],self.list_face_points[380][1],self.list_face_points[385][0],self.list_face_points[385][1])
            dist_hor=distance(self.list_face_points[263][0],self.list_face_points[263][1],self.list_face_points[463][0],self.list_face_points[463][1])
            ear_right=(dist_ver1+dist_ver2)/(2*dist_hor)

            self.right_eye_data.append(ear_right)
            if len(self.right_eye_data)>5:
                self.right_eye_data.pop(0)
            self.right_eye_ratios.append(ear_right)
            if len(self.right_eye_ratios)>2:
                self.right_eye_ratios.pop(0)

            if abs(abs(self.rot_y)-abs(sum(self.rot_y_backup)/3))<0.02 and round(sum(self.right_eye_data)/5,3)*0.86>round(sum(self.right_eye_ratios)/2,3) :
                self.right_eye_clip=True

            if round(sum(self.right_eye_data)/5,5)<round(sum(self.right_eye_ratios)/2,5)*0.876:
                self.right_eye_clip=False
            # if round(sum(self.right_eye_ratios)/2,3)<0.145 :
            #     self.right_eye_clip=True
            # else:
            #     self.right_eye_clip=False


            dist_verl1=distance(self.list_face_points[160][0],self.list_face_points[160][1],self.list_face_points[144][0],self.list_face_points[144][1])
            dist_verl2=distance(self.list_face_points[159][0],self.list_face_points[159][1],self.list_face_points[145][0],self.list_face_points[145][1])
            dist_horl=distance(self.list_face_points[33][0],self.list_face_points[33][1],self.list_face_points[243][0],self.list_face_points[243][1])
            ear_left=(dist_verl1+dist_verl2)/(2*dist_horl)






            self.left_eye_data.append(ear_left)
            if len(self.left_eye_data)>5:
                self.left_eye_data.pop(0)
            self.left_eye_ratios.append(ear_left)
            if len(self.left_eye_ratios)>2:
                self.left_eye_ratios.pop(0)


            if abs(abs(self.rot_y)-abs(sum(self.rot_y_backup)/3))<0.02 and round(sum(self.left_eye_data)/5,3)*0.87>round(sum(self.left_eye_ratios)/2,3) :
                self.left_eye_clip=True
            if round(sum(self.left_eye_data)/5,5)<round(sum(self.left_eye_ratios)/2,5)*0.879:
                self.left_eye_clip=False




            # mouth

            self.m_height=distance(self.list_face_points[12][0],self.list_face_points[12][1],self.list_face_points[15][0],self.list_face_points[15][1])/eyes_dist
            self.m_width=distance(self.list_face_points[191][0],self.list_face_points[191][1],self.list_face_points[407][0],self.list_face_points[407][1])/eyes_dist

            self.m_height+=0.02
            self.m_height=max(self.m_height,0.04)


            new_center=vec2(((self.list_face_points[191][0]+self.list_face_points[407][0])/2)/eyes_dist,((self.list_face_points[15][1]+self.list_face_points[12][1])/2)/eyes_dist)
            center=vec2(self.list_face_points[164][0]/eyes_dist,self.list_face_points[164][1]/eyes_dist)
            self.m_offset=center-new_center
            self.m_offset.y+=0.22


        self.rotation_z=self.create_z_rot_mat(self.rot_z)
        self.rotation_x=self.create_x_rot_mat(self.rot_x)
        self.rotation_y=self.create_y_rot_mat(self.rot_y)
        #self.fTheta+=1*self.dt

        target=ProtoVector(0,0,1)
        matCamRot=self.create_y_rot_mat(self.Yaw)
        self.LookDir=self.MultiplyMatrixVector(target,matCamRot)
        target=self.camera+self.LookDir
        matcamera=self.MatrixPointAt(self.camera,target,self.Up)
        self.matView=self.MatInverse(matcamera)

    def sorter(self,tri):
        dictt={}
        for t in tri:
            dictt[t]=(t.vectors[0].z+t.vectors[1].z+t.vectors[2].z)/3
        items=dictt.items()
        sort=sorted(items,key=lambda x:x[1],reverse=True)
        return [i for i,j in sort]

    def draw_face(self,coords,center,width,face_width):

        rect=pg.Rect(coords.x,coords.y,width,width)
        rect.center=(center.x,center.y)
        surf=pg.Surface((width,width),pg.SRCALPHA)
        pg.draw.rect(surf,RED,rect)
        #surf.set_alpha(80)
        # setting up the face
        if self.blinking:
            if not self.left_eye_clip:
                left_eye=pg.Surface((rect.width/4,rect.width/2))
                left_eye_border=pg.Rect(0,0,left_eye.get_width(),left_eye.get_height())
                left_eye.fill(WHITE)
                pg.draw.rect(left_eye,(1,1,1),left_eye_border,int((left_eye.get_width()*0.15)))
                surf.blit(left_eye,(0+rect.width//8,0+rect.width//8))
            else:
                left_eye=pg.Surface((rect.width/4,rect.width/16))
                left_eye.fill((1,1,1))
                surf.blit(left_eye,(0+rect.width//8,rect.width/2+rect.width/24))
            if not self.right_eye_clip:
                right_eye=pg.Surface((rect.width/4,rect.width/2))
                right_eye_border=pg.Rect(0,0,right_eye.get_width(),right_eye.get_height())
                right_eye.fill(WHITE)
                pg.draw.rect(right_eye,(1,1,1),right_eye_border,int((right_eye.get_width()*0.15)))
                surf.blit(right_eye,(0+(rect.width-rect.width//8-rect.width/4),0+rect.width//8))
            else:
                left_eye=pg.Surface((rect.width/4,rect.width/16))
                left_eye.fill((1,1,1))
                surf.blit(left_eye,(0+(rect.width-rect.width//8-rect.width/4),rect.width/2+rect.width/24))
        else:
            left_eye=pg.Surface((rect.width/4,rect.width/2))
            left_eye_border=pg.Rect(0,0,left_eye.get_width(),left_eye.get_height())
            left_eye.fill(WHITE)
            pg.draw.rect(left_eye,(1,1,1),left_eye_border,int((left_eye.get_width()*0.15)))
            surf.blit(left_eye,(0+rect.width//8,0+rect.width//8))

            right_eye=pg.Surface((rect.width/4,rect.width/2))
            right_eye_border=pg.Rect(0,0,right_eye.get_width(),right_eye.get_height())
            right_eye.fill(WHITE)
            pg.draw.rect(right_eye,(1,1,1),right_eye_border,int((right_eye.get_width()*0.15)))
            surf.blit(right_eye,(0+(rect.width-rect.width//8-rect.width/4),0+rect.width//8))

        mouth=pg.Surface((rect.width/4*(self.m_width*1.75),rect.width/8*(self.m_height*5)))
        mouth_border=pg.Rect(0,0,mouth.get_width(),mouth.get_height())
        mouth.fill(BLACK)
        pg.draw.rect(mouth,(1,1,1),mouth_border,int((mouth.get_width()*0.15)))
        rect_m=mouth.get_rect()
        rect_m.center=(0+width/2+(face_width/2*self.m_offset.x),0+width*3/4+(face_width/2*self.m_offset.y))
        surf.blit(mouth,rect_m)
        #self.screen.blit(surf,rect)
        return surf,rect

    def draw_stripes(self,tristoraster,rect1,rect2,surf_num):
        # draws stripes
            first_tris=[tri for tri in tristoraster if tri.index==rect1][0]
            second_tris=[tri for tri in tristoraster if tri.index==rect2][0]

            topleft_tris=first_tris.vectors[2]-(first_tris.vectors[2]-first_tris.vectors[0])

            toplefts=first_tris.vectors[0]-topleft_tris
            bottomlefts=first_tris.vectors[1]-topleft_tris
            bottomrights=second_tris.vectors[1]-topleft_tris
            toprights=second_tris.vectors[2]-topleft_tris

            surf=self.s_surfs[surf_num]
            surf=pg.transform.scale(surf,self.s_size)
            rects=surf.get_rect()
            center=((first_tris.vectors[2]-(first_tris.vectors[2]-first_tris.vectors[0])/2).x,(first_tris.vectors[2]-(first_tris.vectors[2]-first_tris.vectors[0])/2).y)
            rects.center=center

            rect_toplefts=vec2(rects.topleft)
            tri_toplefts=vec2(first_tris.vectors[0].x,first_tris.vectors[0].y)
            pointers=tri_toplefts-rect_toplefts
            if surf_num==2 or surf_num==6 or surf_num==3 or surf_num==9 or surf_num==13:
                surf=pg.transform.flip(surf,True,False)
                surf.set_colorkey(BLACK)

            if surf_num==4 or surf_num==6 or surf_num==8 or surf_num==10 or surf_num==14:
                surf=pg.transform.rotate(surf,90)
                surf.set_colorkey(BLACK)


            if  surf_num!=14:
                new_imgs=distortion(surf,(rects.width,rects.height),((0,0),(rects.width,0),(0,rects.width),(rects.width,rects.height)),((toplefts.x+pointers.x,toplefts.y+pointers.y),(toprights.x+pointers.x,toprights.y+pointers.y),(bottomlefts.x+pointers.x,bottomlefts.y+pointers.y),(bottomrights.x+pointers.x,bottomrights.y+pointers.y)))
                #
                # m = pg.mask.from_surface(surf)
                # outline = m.outline()
                # if outline:
                    # print(outline)
                    # toplefts_pointer=vec2(rects.center)-vec2(rects.topleft)
                    # ys=[i[1] for i in outline]
                    # ys=[i for i in ys if i!=0]
                    # new_rect=pg.Rect(0,0,rects.width,max(ys)-min(ys))
                    # new_rect.center=(new_imgs.get_width()/2,new_imgs.get_height()/2)
                    # new_new_surf=new_imgs.subsurface(new_rect)
                    # new_new_surf=new_new_surf.copy()
                    #
                    # new_rect.center=rects.center
                    # pg.draw.rect(self.screen,RED,new_rect,1)
                self.screen.blit(new_imgs,rects)



            else:

                new_imgs=distortion(surf,(rects.width,rects.height),((0,0),(rects.width,0),(0,rects.width),(rects.width,rects.height)),((toplefts.x+pointers.x,toplefts.y+pointers.y),(toprights.x+pointers.x,toprights.y+pointers.y),(bottomlefts.x+pointers.x,bottomlefts.y+pointers.y),(bottomrights.x+pointers.x,bottomrights.y+pointers.y)))
                p1=(first_tris.vectors[1]-second_tris.vectors[1]).normalize()
                p2=(second_tris.vectors[1]-first_tris.vectors[2]).normalize()
                p3=(first_tris.vectors[1]-second_tris.vectors[1]).normalize()
                p4=(first_tris.vectors[0]-first_tris.vectors[2]).normalize()
                p5=(first_tris.vectors[0]-first_tris.vectors[1]).normalize()
                p6=(first_tris.vectors[1]-second_tris.vectors[1]).normalize()


                if vec2(p2.x,p2.y).dot(vec2(p1.x,p1.y))<0.95 and vec2(p1.x,p1.y).dot(vec2(p2.x,p2.y))>-0.95 and vec2(p3.x,p3.y).dot(vec2(p4.x,p4.y))<0.95 and vec2(p5.x,p5.y).dot(vec2(p6.x,p6.y))<0.5:

                        self.screen.blit(new_imgs,rects)
                # second_tris.vectors[1].x YELLOW
                # first_tris.vectors[2] green
                # first_tris.vectors[1] blue
                # first_tris.vectors[0] red
    def draw_stripe_exception(self,tristoraster,rect1,rect2,surf_num):
        # draws stripes
            first_tris=[tri for tri in tristoraster if tri.index==rect1][0]
            second_tris=[tri for tri in tristoraster if tri.index==rect2][0]


            # pg.draw.rect(self.screen,YELLOW,(first_tris.vectors[0].x,first_tris.vectors[0].y,3,3))
            # pg.draw.rect(self.screen,GREEN,(first_tris.vectors[1].x,first_tris.vectors[1].y,3,3))
            # pg.draw.rect(self.screen,RED,(first_tris.vectors[2].x,first_tris.vectors[2].y,3,3))
            #
            # pg.draw.rect(self.screen,BLUE,(second_tris.vectors[2].x,second_tris.vectors[2].y,3,3))

            topleft_tris=first_tris.vectors[2]-(first_tris.vectors[2]-first_tris.vectors[0])

            toplefts=first_tris.vectors[0]-topleft_tris
            bottomlefts=first_tris.vectors[1]-topleft_tris
            bottomrights=second_tris.vectors[0]-topleft_tris
            toprights=second_tris.vectors[1]-topleft_tris

            surf=self.s_surfs[surf_num]
            surf=pg.transform.scale(surf,self.s_size)
            rects=surf.get_rect()
            center=((first_tris.vectors[2]-(first_tris.vectors[2]-first_tris.vectors[0])/2).x,(first_tris.vectors[2]-(first_tris.vectors[2]-first_tris.vectors[0])/2).y)
            rects.center=center

            rect_toplefts=vec2(rects.topleft)
            tri_toplefts=vec2(first_tris.vectors[0].x,first_tris.vectors[0].y)
            pointers=tri_toplefts-rect_toplefts

            new_imgs=distortion(surf,(rects.width,rects.height),((0,0),(rects.width,0),(0,rects.width),(rects.width,rects.height)),((toplefts.x+pointers.x,toplefts.y+pointers.y),(toprights.x+pointers.x,toprights.y+pointers.y),(bottomlefts.x+pointers.x,bottomlefts.y+pointers.y),(bottomrights.x+pointers.x,bottomrights.y+pointers.y)))

            self.screen.blit(new_imgs,rects)

    def draw_stripe_exception2(self,tristoraster,rect1,rect2,surf_num):
        # draws stripes
            first_tris=[tri for tri in tristoraster if tri.index==rect1][0]
            second_tris=[tri for tri in tristoraster if tri.index==rect2][0]




            topleft_tris=first_tris.vectors[1]

            toplefts=first_tris.vectors[1]-topleft_tris
            bottomlefts=first_tris.vectors[2]-topleft_tris
            bottomrights=second_tris.vectors[2]-topleft_tris
            toprights=first_tris.vectors[0]-topleft_tris


            surf=self.s_surfs[surf_num]
            surf=pg.transform.scale(surf,self.s_size)
            rects=surf.get_rect()
            center=((first_tris.vectors[2]-(first_tris.vectors[2]-first_tris.vectors[0])/2).x,(first_tris.vectors[2]-(first_tris.vectors[2]-first_tris.vectors[0])/2).y)
            rects.center=center

            rect_toplefts=vec2(rects.topleft)
            tri_toplefts=vec2(first_tris.vectors[1].x,first_tris.vectors[1].y)
            pointers=tri_toplefts-rect_toplefts

            new_imgs=distortion(surf,(rects.width,rects.height),((0,0),(rects.width,0),(0,rects.width),(rects.width,rects.height)),((toplefts.x+pointers.x,toplefts.y+pointers.y),(toprights.x+pointers.x,toprights.y+pointers.y),(bottomlefts.x+pointers.x,bottomlefts.y+pointers.y),(bottomrights.x+pointers.x,bottomrights.y+pointers.y)))

            p1=(toplefts-bottomlefts).normalize()
            p2=(bottomrights-bottomlefts).normalize()
            p3=(toprights-bottomrights).normalize()
            p4=(toprights-toplefts).normalize()

            if vec2(p2.x,p2.y).dot(vec2(p1.x,p1.y))<0.95 and vec2(p4.x,p4.y).dot(vec2(p3.x,p3.y))<0.95:
                self.screen.blit(new_imgs,rects)


    def find_tri_index(self,tris):
            for t in tris:
                center=((t.vectors[2]-(t.vectors[2]-t.vectors[0])/2).x,(t.vectors[2]-(t.vectors[2]-t.vectors[0])/2).y)
                rect=pg.Rect(0,0,20,20)
                rect.center=center
                pg.draw.rect(self.screen,RED,rect,2)
                if rect.collidepoint(pg.mouse.get_pos()):
                    print(t.index)





    def draw(self):

        self.screen.fill(BGCOLOR)
        #self.curr_mesh.tris=self.sorter(self.curr_mesh.tris)



        tristoraster=[]
        for n,tri in enumerate(self.curr_mesh.tris):

            # rotation

            #triRotZ=Triangle([self.MultiplyMatrixVector(tri.vectors[0],self.rotation_z),self.MultiplyMatrixVector(tri.vectors[1],self.rotation_z),self.MultiplyMatrixVector(tri.vectors[2],self.rotation_z)])
            triRotZ=tri.copy()
            triRotZ.vectors[0]=self.MultiplyMatrixVector(tri.vectors[0],self.rotation_z)
            triRotZ.vectors[1]=self.MultiplyMatrixVector(tri.vectors[1],self.rotation_z)
            triRotZ.vectors[2]=self.MultiplyMatrixVector(tri.vectors[2],self.rotation_z)

            #triRotX=Triangle([self.MultiplyMatrixVector(triRotZ.vectors[0],self.rotation_x),self.MultiplyMatrixVector(triRotZ.vectors[1],self.rotation_x),self.MultiplyMatrixVector(triRotZ.vectors[2],self.rotation_x)])
            triRotX=triRotZ.copy()
            triRotX.vectors[0]=self.MultiplyMatrixVector(triRotZ.vectors[0],self.rotation_x)
            triRotX.vectors[1]=self.MultiplyMatrixVector(triRotZ.vectors[1],self.rotation_x)
            triRotX.vectors[2]=self.MultiplyMatrixVector(triRotZ.vectors[2],self.rotation_x)

            #triRotY=Triangle([self.MultiplyMatrixVector(triRotX.vectors[0],self.rotation_y),self.MultiplyMatrixVector(triRotX.vectors[1],self.rotation_y),self.MultiplyMatrixVector(triRotX.vectors[2],self.rotation_y)])
            triRotY=triRotX.copy()
            triRotY.vectors[0]=self.MultiplyMatrixVector(triRotX.vectors[0],self.rotation_y)
            triRotY.vectors[1]=self.MultiplyMatrixVector(triRotX.vectors[1],self.rotation_y)
            triRotY.vectors[2]=self.MultiplyMatrixVector(triRotX.vectors[2],self.rotation_y)

            # translation

            triTranslated= triRotY.copy()
            triTranslated.vectors[0].z+=6
            triTranslated.vectors[1].z+=6
            triTranslated.vectors[2].z+=6
            # triTranslated.vectors[0].x-=0.5
            # triTranslated.vectors[1].x-=0.5
            # triTranslated.vectors[2].x-=0.5

            # cross_prod
            line1=triTranslated.vectors[1]-triTranslated.vectors[0]
            line2=triTranslated.vectors[2]-triTranslated.vectors[0]
            normal=line1.cross(line2)

            normal=normal.normalize()


            if normal.dot(ProtoVector(triTranslated.vectors[0].x-self.camera.x,triTranslated.vectors[0].y-self.camera.y,triTranslated.vectors[0].z-self.camera.z))<0:
                tri.visible=True
                tri.normal_dp= normal.dot(ProtoVector(triTranslated.vectors[0].x-self.camera.x,triTranslated.vectors[0].y-self.camera.y,triTranslated.vectors[0].z-self.camera.z))


            if triTranslated.visible:
                # illumination
                #triTranslated.normal_dp=normal.dot(ProtoVector(triTranslated.vectors[0].x-self.camera.x,triTranslated.vectors[0].y-self.camera.y,triTranslated.vectors[0].z-self.camera.z))
                light_dir=ProtoVector(0,0,-1)
                light_dir.normalize()

                light_dp=normal.dot(light_dir)
                #triViewed=Triangle([self.MultiplyMatrixVector(triTranslated.vectors[0],self.matView) ,self.MultiplyMatrixVector(triTranslated.vectors[1],self.matView),self.MultiplyMatrixVector(triTranslated.vectors[2],self.matView)])
                triViewed=triTranslated.copy()
                triViewed.vectors[0]=self.MultiplyMatrixVector(triTranslated.vectors[0],self.matView)
                triViewed.vectors[1]=self.MultiplyMatrixVector(triTranslated.vectors[1],self.matView)
                triViewed.vectors[2]=self.MultiplyMatrixVector(triTranslated.vectors[2],self.matView)

                # clipping agains a near plain


                count_clipped,clipped=self.Triange_ClipAgainstPlane(ProtoVector(0.,0.,0.1),ProtoVector(0.,0.,1.),triViewed)

                clipped=[t for t in clipped if t]
                for i in range(count_clipped):
                    #triprojected=Triangle([self.MultiplyMatrixVector(clipped[i].vectors[0],self.matProj) ,self.MultiplyMatrixVector(clipped[i].vectors[1],self.matProj),self.MultiplyMatrixVector(clipped[i].vectors[2],self.matProj)])

                    triprojected=clipped[i].copy()
                    triprojected.vectors[0]=self.MultiplyMatrixVector(clipped[i].vectors[0],self.matProj)
                    triprojected.vectors[1]=self.MultiplyMatrixVector(clipped[i].vectors[1],self.matProj)
                    triprojected.vectors[2]=self.MultiplyMatrixVector(clipped[i].vectors[2],self.matProj)

                    # division by w(z)
                    triprojected.vectors[0]/=triprojected.vectors[0].w
                    triprojected.vectors[1]/=triprojected.vectors[1].w
                    triprojected.vectors[2]/=triprojected.vectors[2].w


                    # shifting points
                    offsetvector=ProtoVector(1,1,0)
                    triprojected.vectors[0]+=offsetvector
                    triprojected.vectors[1]+=offsetvector
                    triprojected.vectors[2]+=offsetvector

                    scale=ProtoVector(0.5*WIDTH,0.5*HEIGHT,1)
                    triprojected.vectors[0]*=scale
                    triprojected.vectors[1]*=scale
                    triprojected.vectors[2]*=scale

                    triprojected.light_dp=light_dp
                    tri.light_dp=light_dp

                    triprojected.back_up()

                    tristoraster.append(triprojected)







        tristoraster=self.sorter(tristoraster)




        # clipping against screen
        for plane in range(4):

            if plane==0:
                new_queue=[]

                for tri in tristoraster:
                    triToAdd,clipped=self.Triange_ClipAgainstPlane(ProtoVector(0.,0.,0.),ProtoVector(0.,1.,0.),tri)
                    new_queue+=clipped
                new_queue=[t for t in new_queue if t]
                tristoraster=new_queue

            if plane==1:

                new_queue=[]

                for tri in tristoraster:
                    triToAdd,clipped=self.Triange_ClipAgainstPlane(ProtoVector(0.,self.screen.get_height()-1.,0.),ProtoVector(0.,-1.,0.),tri)
                    new_queue+=clipped
                new_queue=[t for t in new_queue if t]
                tristoraster=new_queue

            if plane==2:
                new_queue=[]

                for tri in tristoraster:
                    triToAdd,clipped=self.Triange_ClipAgainstPlane(ProtoVector(0.,0.,0.),ProtoVector(1.,0.,0.),tri)
                    new_queue+=clipped
                new_queue=[t for t in new_queue if t]
                tristoraster=new_queue

            if plane==3:
                new_queue=[]

                for tri in tristoraster:
                    triToAdd,clipped=self.Triange_ClipAgainstPlane(ProtoVector(self.screen.get_width()-1.,0.,0.),ProtoVector(-1.,0.,0.),tri)
                    new_queue+=clipped
                new_queue=[t for t in new_queue if t]
                tristoraster=new_queue




        # smoothing


        for tri in tristoraster:
            tri.mean()






        # rastering

        for triprojected in  tristoraster:

            grey=int(255*triprojected.light_dp)
            r=int(72*triprojected.light_dp)
            g=int(61*triprojected.light_dp)
            b=int(139*triprojected.light_dp)
            if r<0:
                r=0
            if g<0:
                g=0
            if b<0:
                b=0

            pg.draw.polygon(self.screen,(r,g,b),((triprojected.vectors[0].x,triprojected.vectors[0].y),(triprojected.vectors[1].x,triprojected.vectors[1].y),(triprojected.vectors[2].x,triprojected.vectors[2].y)))
            #pg.draw.polygon(self.screen,(grey,grey,grey),((triprojected.vectors[0].x,triprojected.vectors[0].y),(triprojected.vectors[1].x,triprojected.vectors[1].y),(triprojected.vectors[2].x,triprojected.vectors[2].y)))
            # if triprojected.index%2==0:
            #     self.draw_text(str(triprojected.index),self.font,30,BLACK,triprojected.vectors[0].x-20,triprojected.vectors[0].y,)
            #
            # else:
            #     self.draw_text(str(triprojected.index),self.font,30,GREEN,triprojected.vectors[0].x-20,triprojected.vectors[0].y,)

            if triprojected.index==59 or triprojected.index==56 or triprojected.index==58 or triprojected.index==57 or triprojected.index==19 or triprojected.index==55:
                if self.rot_x>0.2:
                    pg.draw.polygon(self.screen,self.default_color,((triprojected.vectors[0].x,triprojected.vectors[0].y),(triprojected.vectors[1].x,triprojected.vectors[1].y),(triprojected.vectors[2].x,triprojected.vectors[2].y)))
            # if triprojected.index==23 or triprojected.index==22:
            #     pg.draw.polygon(self.screen,RED,((triprojected.vectors[0].x,triprojected.vectors[0].y),(triprojected.vectors[1].x,triprojected.vectors[1].y),(triprojected.vectors[2].x,triprojected.vectors[2].y)))
            #pg.draw.polygon(self.screen,BLACK,((triprojected.vectors[0].x,triprojected.vectors[0].y),(triprojected.vectors[1].x,triprojected.vectors[1].y),(triprojected.vectors[2].x,triprojected.vectors[2].y)),1)





        if tristoraster:

            #draw stripes
            if self.stripes.current_plane==5 and self.rot_y>0.2:
                try:

                    self.draw_stripe_exception(tristoraster,27,1,5)
                    self.draw_stripes(tristoraster,25,26,10)
                    self.draw_stripes(tristoraster,28,29,11)
                except:
                    pass
            if self.stripes.current_plane==0 and self.rot_y>0.2:
                try:

                    self.draw_stripe_exception(tristoraster,27,1,5)
                    self.draw_stripes(tristoraster,25,26,10)
                    self.draw_stripes(tristoraster,28,29,11)
                    self.draw_stripes(tristoraster,34,6,2)
                    self.draw_stripes(tristoraster,33,5,6)
                    self.draw_stripes(tristoraster,32,4,7)
                except:
                    pass


            if self.stripes.current_plane==1:
                self.draw_stripes(tristoraster,34,6,2)
                self.draw_stripes(tristoraster,33,5,6)
                self.draw_stripes(tristoraster,32,4,7)
                self.draw_stripes(tristoraster,30,2,0)
                self.draw_stripes(tristoraster,31,3,1)
                self.draw_stripes(tristoraster,38,10,3)
            if self.stripes.current_plane==2:
                self.draw_stripes(tristoraster,30,2,0)
                self.draw_stripes(tristoraster,31,3,1)
                self.draw_stripes(tristoraster,38,10,3)
                self.draw_stripes(tristoraster,37,9,4)
                self.draw_stripe_exception2(tristoraster,35,7,8)
                self.draw_stripes(tristoraster,36,8,9)
            if self.stripes.current_plane==3 :
                self.draw_stripes(tristoraster,37,9,4)
                self.draw_stripe_exception2(tristoraster,35,7,8)
                self.draw_stripes(tristoraster,36,8,9)
                if self.rot_y<-0.15:
                    try:
                        self.draw_stripe_exception(tristoraster,21,24,12)
                        self.draw_stripes(tristoraster,0,20,13)
                        self.draw_stripes(tristoraster,23,22,14)
                    except:
                        pass

            if self.stripes.current_plane==4 and self.rot_y<-0.2:
                try:
                    self.draw_stripe_exception(tristoraster,21,24,12)
                    self.draw_stripes(tristoraster,0,20,13)
                    self.draw_stripes(tristoraster,23,22,14)
                except:
                    pass

            #draws face
            first_tri=[tri for tri in tristoraster if tri.index==30][0]
            second_tri=[tri for tri in tristoraster if tri.index==2][0]

            topleft_tri=first_tri.vectors[2]-(first_tri.vectors[2]-first_tri.vectors[0])
            topleft=first_tri.vectors[0]-topleft_tri
            bottomleft=first_tri.vectors[1]-topleft_tri
            bottomright=second_tri.vectors[1]-topleft_tri
            topright=second_tri.vectors[2]-topleft_tri
            img,rect=self.draw_face(first_tri.vectors[0],first_tri.vectors[2]-(first_tri.vectors[2]-first_tri.vectors[0])/2,(first_tri.vectors[2]-first_tri.vectors[0]).length(),(topright-topleft).length())
            self.s_size=(rect.width,rect.height)

            # rect_topleft_tri_topleft pointer vector
            rect_topleft=vec2(rect.topleft)
            tri_topleft=vec2(first_tri.vectors[0].x,first_tri.vectors[0].y)
            pointer=tri_topleft-rect_topleft


            new_img=distortion(img,(rect.width,rect.height),((0,0),(rect.width,0),(0,rect.width),(rect.width,rect.height)),((topleft.x+pointer.x,topleft.y+pointer.y),(topright.x+pointer.x,topright.y+pointer.y),(bottomleft.x+pointer.x,bottomleft.y+pointer.y),(bottomright.x+pointer.x,bottomright.y+pointer.y)))
            self.screen.blit(new_img,rect)


        # fps
        f=self.draw_text(str(int(self.clock.get_fps())), self.font, 40, WHITE, 50, 50, align="center")
        self.draw_text('B - BLINKING'+':'+str(self.blinking), self.font, 30, WHITE, 50, f.bottom+20, )
        pg.display.flip()

    def events(self):
        # catch all events here
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.quit()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    self.quit()
                if event.key == pg.K_UP:
                    self.camera.y+=40*self.dt
                if event.key == pg.K_DOWN:
                    self.camera.y-=40*self.dt
                if event.key == pg.K_LEFT:
                    self.camera.x+=40*self.dt
                if event.key == pg.K_RIGHT:
                    self.camera.x-=40*self.dt

                if event.key == pg.K_a:
                    self.Yaw-=5*self.dt

                if event.key == pg.K_d:
                    self.Yaw+=5*self.dt

                if event.key == pg.K_w:
                    self.camera+=self.LookDir*40*self.dt

                if event.key == pg.K_s:
                    self.camera-=self.LookDir*40*self.dt
                if event.key == pg.K_b:
                    self.blinking=not self.blinking

                



# create the game object
g = Game()
g.new()
g.run()
