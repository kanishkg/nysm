import numpy as np


def make_rect(x1,y1,x2,y2):
    rect = np.zeros((361,181))
    if x2<x1:
        rect[:int(x2*360),int(y1*180):int(y2*180)] = 1
        rect[int(x1*360):,int(y1*180):int(y2*180)] = 1
        return rect
    rect[int(x1*360):int(x2*360),int(y1*180):int(y2*180)] = 1
    return rect

def get_hit_rate(out,batch,a):
    hit_rate = []
    out[1][:,:] = np.sqrt(np.exp(out[1][:,:]))
    out[1][:,0] = np.clip(out[1][:,0]/(2*np.pi),None,0.5)
    out[1][:,1] = np.clip(out[1][:,1]/(np.pi),None,2.0/3.0)
    for i in range (batch['target'].shape[0]):
        f1=0
        x1 = out[0][i,0]-a*out[1][i,0]/2
        y1 = out[0][i,1]-a*out[1][i,1]/2
        x2 = out[0][i,0]+a*out[1][i,0]/2
        y2 = out[0][i,1]+a*out[1][i,1]/2
        orect = (x1,y1,x2,y2)
        if x1<0:
            x1 = 1+x1
        if y1<0:
            y1 = 0.0
        if x2>1:
            x2 = x2-1
        if y2>1:
            y2 = 1.0
        r1 = make_rect(x1,y1,x2,y2)
        h=0
        for j in range(15):
            f2 =0
            x3 = batch['target'][i,j]-1/6.0
            y3 = batch['target'][i,15+j]-0.25
            x4 = batch['target'][i,j]+1/6.0
            y4 = batch['target'][i,15+j]+0.25
            orect2 = (x3,y3,x4,y4)
            if x3<0:
                x3 = 1+x3
            if y3<0:
                y3 = 0.0
            if x4>1:
                x4 = x4-1
            if y4>1:
                y4 = 1.0
            r2 = make_rect(x3,y3,x4,y4)

            A_sent = np.sum(r2)
            A = np.sum(np.multiply(r1,r2))
            if A>A_sent:
                print ("How",A,A_sent,f1,f2,x1,x2,x3,x4,y1,y2,y3,y4,[x5,y5,x6,y6],orect,orect2,)
            h += float(A)/float(A_sent)
        hit_rate.append(h/15)
    return hit_rate
