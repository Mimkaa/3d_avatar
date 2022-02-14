import pygame as pg
vec=pg.Vector2
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
PURPLE1=(164,123,249)
PURPLE2=(123,129,249)
PURPLE3=(153,149,251)

class Running_Stripes:
    def __init__(self,planes,colors):
        self.planes=planes
        self.colors=colors
        self.current_plane=0
        self.current_plane1=1
        self.current_pos=vec(0,0)
        self.current_pos1=vec(-self.planes[self.current_plane].get_width()*2,0)

        self.stripe=Stripe(self.colors,self.planes[0].get_size())
        self.stripe1=pg.Surface((10,10))
        self.x_move=0
        self.x_move1=0
        self.rect=self.planes[self.current_plane].get_rect()
        self.stripe_rect=self.stripe.image.get_rect()
        self.index_plane_dict={0:[self.planes[5],self.planes[10],self.planes[11]],1:[self.planes[2],self.planes[6],self.planes[7]],2:[self.planes[0],self.planes[1],self.planes[3]],3:[self.planes[4],self.planes[8],self.planes[9]],4:[self.planes[12],self.planes[13],self.planes[14]],5:[]}
    def update(self,dt):

        self.x_move=0
        self.x_move+=300*dt
        self.current_pos.x+=self.x_move


        if self.current_pos.x>self.planes[0].get_width() :
            self.current_plane+=1
            self.current_pos=vec(0,0)
            if self.current_plane>len(self.index_plane_dict.keys())-1:
                self.current_plane=0
                self.current_pos=vec(0,0)

    def render(self):
        for plane in self.planes:
            plane.fill(BLACK)

        if self.current_pos.x>=0:
            width=self.current_pos.x
            try:
                self.stripe1=self.stripe.image.subsurface(self.stripe.image.get_width()-width,0,width,self.stripe.image.get_height())
            except:
                self.stripe1=pg.Surface((10,10))

            if self.current_plane<len(self.index_plane_dict.keys())-1:

                    for plane in self.index_plane_dict[self.current_plane+1]:

                            plane.blit(self.stripe1,(0,0))

            else:
                for plane in self.index_plane_dict[0]:
                            plane.blit(self.stripe1,(0,0))

        for key,value in self.index_plane_dict.items():
            if key==self.current_plane:
                for plane in value:
                    plane.fill(BLACK)
                    plane.blit(self.stripe.image,self.current_pos)

        return self.planes


class Stripe:
    def __init__(self,colours,size):
        self.image=pg.Surface(size,pg.SRCALPHA)
        self.image2=pg.Surface(size,pg.SRCALPHA)
        self.colours=colours
        self.rect=self.image.get_rect()
        for i in range(len(self.colours)):
            pg.draw.polygon(self.image,self.colours[i],((self.rect.left+i*((self.rect.width/2)/len(self.colours)),self.rect.top),
                                                        (self.rect.left+i*(self.rect.width/2)/len(self.colours)+(self.rect.width/2)/len(self.colours),self.rect.top),
                                                        (self.rect.left+i*(self.rect.width/2)/len(self.colours)+(self.rect.width/2)/len(self.colours),self.rect.bottom),
                                                        (self.rect.left+i*((self.rect.width/2)/len(self.colours)),self.rect.bottom)
                                                        ))
        for i in range(len(self.colours)):
            pg.draw.polygon(self.image,self.colours[i],((self.rect.left+i*((self.rect.width*2)/len(self.colours)),self.rect.top),
                                                        (self.rect.left+i*(self.rect.width*2)/len(self.colours)+(self.rect.width*2)/len(self.colours),self.rect.top),
                                                        (self.rect.left+i*(self.rect.width*2)/len(self.colours)+(self.rect.width*2)/len(self.colours),self.rect.bottom),
                                                        (self.rect.left+i*((self.rect.width*2)/len(self.colours)),self.rect.bottom)
                                                        ))

