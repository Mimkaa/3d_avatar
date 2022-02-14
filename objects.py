from math import sqrt
import pygame as pg
from os import path

def numeric_custom(strr):
    list=[str(i) for i in range(10)]
    list.append('.')
    list.append('-')
    is_num=False
    for n in list:
        for l in strr:
            if l==n:
                is_num=True

    return is_num

class ProtoVector:
    def __init__(self,a,b=None,s=None,w=1):
        if  type(a)!=type(self) and b==None:

            x,y,z,p=a[0],a[1],a[2],w
        elif type(a)==type(self):
            x,y,z,p=a.x,a.y,a.z,a.w
        else:
            x,y,z,p=a,b,s,w

        self.x=float(x)
        self.y=float(y)

        self.z=float(z)
        self.w=p

    def __add__(self,other):

        if type(other)==type(self):
            return ProtoVector(self.x+other.x,self.y+other.y,self.z+other.z)
        else:
            return ProtoVector(self.x+other,self.y+other,self.z+other)
    def __sub__(self,other):
        if type(other)==type(self):
            return ProtoVector(self.x-other.x,self.y-other.y,self.z-other.z)
        else:
            return ProtoVector(self.x-other,self.y-other,self.z-other)
    def __mul__(self,other):
        if type(other)==type(self):
            return ProtoVector(self.x*other.x,self.y*other.y,self.z*other.z)
        else:
            return ProtoVector(self.x*other,self.y*other,self.z*other)
    def __truediv__(self,other):
        if type(other)==type(self):
            return ProtoVector(self.x/other.x,self.y/other.y,self.z/other.z)
        else:
            return ProtoVector(self.x/other,self.y/other,self.z/other)
    def __floordiv__(self,other):
        if type(other)==type(self):
            return ProtoVector(self.x//other.x,self.y//other.y,self.z//other.z)
        else:
            return ProtoVector(self.x//other,self.y//other,self.z//other)
    def __iadd__(self, other):
        if type(other)==type(self):
            self.x+=other.x
            self.y+=other.y
            self.z+=other.z
        else:
            self.x+=other
            self.y+=other
            self.z+=other
        return self
    def __itruediv__(self, other):
        if type(other)==type(self):
            self.x/=other.x
            self.y/=other.y
            self.z/=other.z
        else:
            self.x/=other
            self.y/=other
            self.z/=other
        return self
    def __ifloordiv__(self, other):
        if type(other)==type(self):
            self.x//=other.x
            self.y//=other.y
            self.z//=other.z
        else:
            self.x//=other
            self.y//=other
            self.z//=other
        return self
    def __imul__(self, other):
        if type(other)==type(self):
            self.x*=other.x
            self.y*=other.y
            self.z*=other.z
        else:
            self.x*=other
            self.y*=other
            self.z*=other
        return self

    def length(self):
        return sqrt(self.x**2+self.y**2+self.z**2)
    def normalize(self):

        return ProtoVector(self.x/self.length(),self.y/self.length(),self.z/self.length())
    def __repr__(self):
        return f"v({self.x},{self.y},{self.z},{self.w})"

    def cross(self,other):
        return ProtoVector(self.y*other.z-self.z*other.y,self.z*other.x-self.x*other.z,self.x*other.y-self.y*other.x)

    def dot(self,other):
        return self.x*other.x+self.y*other.y+self.z*other.z

    def copy(self):
        return ProtoVector(self.x,self.y,self.z,w=self.w)
    def scale_to(self,val):
        new=self*(val/self.length())
        return new

class Triangle:
    def __init__(self,vectors):
        self.vectors=[ProtoVector(vector) for vector in vectors]
        self.light_dp=0
        self.history=[]
        self.index=0
        self.visible=False
        self.normal_dp=0
    def copy(self):
        tri=Triangle(self.vectors)
        tri.light_dp=self.light_dp
        tri.index=self.index
        tri.history=self.history
        tri.visible=self.visible
        tri.normal_dp=self.normal_dp
        return tri
    def back_up(self):
        self.history.append(self.vectors)
        if len(self.history)>5:
            self.history.pop(0)
    def mean(self):
        #if len(self.history)>=5:
            sum_vec_f=ProtoVector(0,0,0)
            sum_vec_s=ProtoVector(0,0,0)
            sum_vec_t=ProtoVector(0,0,0)
            for j in range(len(self.history)):
                sum_vec_f+=self.history[j][0]
                sum_vec_s+=self.history[j][1]
                sum_vec_t+=self.history[j][2]
            sum_vec_f=sum_vec_f/(len(self.history))
            sum_vec_s=sum_vec_s/(len(self.history))
            sum_vec_t=sum_vec_t/(len(self.history))
            self.vectors[0]=sum_vec_f
            self.vectors[1]=sum_vec_s
            self.vectors[2]=sum_vec_t








# tris - triangles
class Mesh:
    def __init__(self,triangles):
        self.tris=[Triangle(tri) for tri in triangles]
    def LoadFromObjectFile(self,name):
        self.tris.clear()
        list_ver=[]
        list_fs=[]
        with open(path.join(name), 'r') as f:
            lines=f.readlines()
            for n,line in enumerate(lines):



                if line[0]=='v' and line[1]!='t' and line[1]!='n':
                    list_ver.append([float(i.strip('\n')) for i in line[1:].split(' ') if numeric_custom(i)])
                if line[0]=='f':
                    if not'/' in line:
                        list_fs.append([int(i.strip('\n')) for i in line[1:].split(' ') if numeric_custom(i)])
                    else:
                        fss=line[1:].split(' ')
                        listt=[int(i.strip('\n')) for i in fss[1].split('/') if numeric_custom(i)]
                        list_fs.append(listt)

        tris=[]
        for fs in list_fs:
            temp=[]
            for n in fs:
                temp.append(list_ver[n-1])
            tris.append(temp)

        self.tris=[Triangle(tri) for tri in tris]
    # def LoadFromObjectFile(self,name,haveAtex=False):
    #     list_ver=[]
    #     list_fs=[]
    #     list_ver_tex=[]
    #
    #
    #     with open(path.join(name), 'r') as f:
    #         lines=f.readlines()
    #
    #         for line in lines:
    #
    #             if line[0]=='v' :
    #                 if line[1]!='t':
    #                     list_ver.append([float(i.strip('\n')) for i in line[1:].split(' ') if numeric_custom(i)  ])
    #                 else:
    #                     list_ver_tex.append([float(i.strip('\n')) for i in line[1:].split(' ') if numeric_custom(i)  ])
    #
    #             if not  haveAtex:
    #                 if line[0]=='f':
    #                     list_fs.append([int(i.strip('\n')) for i in line[1:].split(' ') if i!='f' and i!='' ])
    #             else:
    #                 if line[0]=='f':
    #                     list_vertex=[i.strip('\n') for i in line[1:].split(' ') if i!='f' and i!='' ]
    #                     list_vertex=[i for i in list_vertex if i!='']
    #                     splitted=[tuple(i.split('/')) for i in list_vertex]
    #                     numbers=[(int(i),int(j)) for i,j in splitted]
    #                     text=[j for i,j in numbers]
    #                     ver=[i for i,j in numbers]
    #                     list_fs.append((ver,text))
    #
    #     tri_list=[]
    #     tex_list=[]
    #     tex_ver_tirs=[]
    #     if haveAtex:
    #         for fs in list_fs:
    #             listt=[]
    #             listt2=[]
    #
    #             for v in tuple(fs[0]):
    #
    #                 listt.append(list_ver[v-1])
    #             for t in tuple(fs[1]):
    #                 listt2.append(list_ver_tex[t-1])
    #             tri_list.append(listt)
    #             tex_list.append(listt2)
    #         tex_ver_tirs=list(zip(tri_list,tex_list))
    #
    #         self.tris=[Triangle(tri,tex_tri) for tri,tex_tri in tex_ver_tirs]
    #     else:
    #         for fs in list_fs:
    #             listt=[]
    #
    #
    #             for v in tuple(fs):
    #
    #                 listt.append(list_ver[v-1])
    #
    #             tri_list.append(listt)
    #
    #
    #
    #         self.tris=[Triangle(tri) for tri in tri_list]