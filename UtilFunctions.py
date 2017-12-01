import numpy as np
import cv2
import math
import sys
import tensorflow as tf
# import plot
np.set_printoptions(threshold=np.nan)

def maxdiff(im):
    h,w = im.shape
    lower_lmt = -1
    upper_lmt = h
    for y in range(0,h):
        for x in range(0,w):
            if(im[y][x]==0):
                lower_lmt = y
                break
        if(lower_lmt != -1):
            break
    for y in range(0,h):
        for x in range(0,w):
            if(im[h-y-1][x]==0):
                upper_lmt = h-y-1
                break
        if(upper_lmt != h):
            break
    return lower_lmt,upper_lmt



def dist(x1,y1,x2,y2):
    return ((x2-x1)**2 + (y2-y1)**2)**(1/2)

def distance(box1,x2,y2):
    x1 = box1[0]+(box1[2]/2)
    y1 = box1[1]+(box1[3]/2)
    return dist(x1,y1,x2,y2)

def findConnection(components,x1,y1,x2,y2):
    c1=-1
    c2=-1
    m=10000
    for i in range(len(components)):
        d = distance(components[i],x1,y1)
        if(d < m):
            m = d
            c1=i
    m=10000
    for i in range(len(components)):
        d = distance(components[i],x2,y2)
        if(d < m and i != c1):
            m = d
            c2=i
    if(c1>100 or c2>100):
        return(-1,-1)
    return (c1,c2)

def show(image,name="image"):
    cv2.imshow(name,image)
    cv2.waitKey(0)

def findlines(img,mul,x=10):
    thresh = img.copy()
    comp = np.zeros(img.shape,np.uint8)
    i=0
    while (i+mul<thresh.shape[0]):
        j=0
        while (j+mul<thresh.shape[1]):
            temp = []
            for i1 in range (i,i+mul):
                temp1 = []
                for j1 in range (j,j+mul):
                    #print(i1,j1)
                    temp1.append(thresh[i1][j1])
                temp.append(temp1)
            temp=np.array(temp)
            _,contours,h1 = cv2.findContours(temp,1,2)
            
            if (len(contours)>1):
                for i1 in range (i,i+mul):
                    for j1 in range (j,j+mul):
                        # thresh[i1][j1]=0
                        comp[i1][j1]=255
            j+=x
        i+=x
    return comp

def rotateImage(image, angle):
    if (angle<5 and angle>-5):
        return image
    image_center = tuple(np.array(image.shape)/2)
    rot_mat = cv2.getRotationMatrix2D(image_center,angle,1.0)
    x,y = image.shape
    result = cv2.warpAffine(image, rot_mat, (y,x),flags=cv2.INTER_LINEAR)
    return result

def getComponents(image, components):
    count = 0
    CompImages = []
    for component in components:
        count+=1
        x,y,w,h = component
        a=0
        b=0
        for i in range(x,x+w):
            if(image[y][i]==0):
                a=1
                break;
            #image[y][y]==0
        for i in range(x,x+w):
            if(image[y+3][i]==0):
                b=1
                break;
##            image[i][y+h]==0
        if(a==0 or b==0):
            img = np.zeros((h,w),np.uint8)
            for X in range(0,h):
                for Y in range(0,w):
                    img[X][Y] = image[y+X][x+Y]
        else:
            img = np.zeros((w,h),np.uint8)
            for X in range(0,h):
                for Y in range(0,w):
                    img[Y][X] = image[y+X][x+Y]
            e = h
            h = w
            w = e
        for i in range(0,h):
            if(img[i][0]==0):
                a=i
                break;
            #image[y][y]==0
        for i in range(0,h):
            if(img[i][w-1]==0):
                b=i
                break;
        angle = math.atan((b-a)/w)
        angle *= (180/math.pi)
        img = rotateImage(img,angle)
        x,y = img.shape
        for i in range (0,y):
            j=0
            while (j<x and img[j][i]<254):
                img[j][i]=255
                j=j+1
            j=x-1
            while (j>0 and img[j][i]<254):
                img[j][i]=255
                j=j-1
        ret,BW = cv2.threshold(img,225,255,0)
        CompImages.append(BW)
    return CompImages

def findcomponents(i):
    X_test = []
    orig=i.copy()
    img1=i.copy()   
    img2=i.copy()
    (wm,hm)=img1.shape
    ret,thresh = cv2.threshold(img1,150,255,1)   ## skeletonization needs white on black so set to 1
    ret,BW = cv2.threshold(img1,150,255,0)      # gives Black on white
    img=thresh.copy()
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False
    skel = np.zeros(img.shape,np.uint8)
    size = np.size(img)
    while(done==0):
        eroded = cv2.erode(img,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(img,temp)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()
        zeros = size - cv2.countNonZero(img)
        if zeros==size:
            done = True

    final=thresh.copy()
    sqsize = 20
    shift = 5
    comp=findlines(thresh,sqsize,shift)
##    show(comp,"comp")
    components = []
    wires = []
    wires_img=i.copy()
    _,contours,h1 = cv2.findContours(comp,1 ,2)
    # print("components")
    for cont in contours:
        (x,y,w,h)= cv2.boundingRect(cont)
        if(w<(sqsize*2) or h<(sqsize*2)):
            continue
        x-=5
        y-=5
        w+=10
        h+=10
        components.append((x,y,w,h))
        cv2.rectangle(orig,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.rectangle(final,(x,y),(x+w,y+h),0,-1)

    CompImages = getComponents(BW, components)
    for i in range (0,len(CompImages)):
        tempimage=CompImages[i]
        tempimage = cv2.resize(tempimage, (100, 40))
        CompImages[i] = tempimage
        
        ret,thresh1 = cv2.threshold(tempimage,225,255,cv2.THRESH_BINARY)
        temp=[]
        flag1=0
        flag2=0
        lower,upper = maxdiff(thresh1)
        w,h = thresh1.shape
        for i in range (0,h):
            temp1=1
            temp2=0
            for j in range (0,w):
                if (thresh1[j][i]==0):
                    temp1=j
                    for j1 in range (0,w):
                        if (thresh1[w-1-j1][i]==0):
                            temp2=w-1-j1
                            break
                    break
            if(temp1==1 and temp2==0):
                temp.append(-100)
            else :
                temp.append(int((abs(temp1-temp2)*100)/abs(upper-lower)))
        i=0
        while (temp[i]==-100 and i<99):
            temp[i]=0
            i=i+1
            if (temp[i]!=-100):
                break
        i=99
        while (temp[i]==-100 and i>0):
            temp[i]=0
            i-=1
            if (temp[i]!=-100):
                break
        X_test.append(temp)

    _,contours,h1 = cv2.findContours(final,1 ,2)
    for cont in contours:
        (x,y,w,h)= cv2.boundingRect(cont)
        wires.append(cont)
        cv2.rectangle(wires_img,(x,y),(x+w,y+h),(0,0,255),2)

    # connections list contains connection in tuple of 2, format:(index of comp1, index of comp2) 
    connections = []

    for i in range(len(wires)):
        x1 = wires[i][0][0][0]
        y1 = wires[i][0][0][1]
        x2 = wires[i][int(len(wires[i])/2)][0][0]
        y2 = wires[i][int(len(wires[i])/2)][0][1]
        c1,c2 = findConnection(components,x1,y1,x2,y2);
        connections.append((c1,c2))

    return X_test,connections,components
