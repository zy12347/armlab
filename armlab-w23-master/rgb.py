import cv2
import numpy as np
import matplotlib.pyplot as plt

def colors(hsv):
    colorbar = ['g','y','r','r','o','b','p']
    color_bounds_lower = np.array([[35,43,46],[26,43,46],[0,43,46],[156,43,46],[11,43,46],[100,43,46],[125,43,46]])
    color_bounds_upper = np.array([[77,255,255],[34,255,255],[10,255,255],[180,255,255],[25,255,255],[124,255,255],[155,255,255]])
    index = 0
    low = hsv>=color_bounds_lower
    low_ = np.sum(low,axis=1)
    up = hsv <= color_bounds_upper
    up_ = np.sum(up, axis=1)
    whole = low_+up_
    index = np.argmax(whole)
    return colorbar[index]

def main():
    im = cv2.imread("tgresgd.png")
    location = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    print(im.shape)

    image = cv2.imread("dawsd.png")
    plt.imshow(image)
    plt.show()
    print(image.shape)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    print(hsv.shape)
    print(hsv[0,0,:])
    print()
    ret, binary = cv2.threshold(location,127,255,0)
    kernel = np.ones((5, 5), 'uint8')
    binary = cv2.dilate(binary, kernel)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    print(len(contours))
    #print(contours)

    color_bounds_lower = np.array([[35,43,46],[26,43,46],[0,43,46],[156,43,46],[11,43,46],[100,43,46],[125,43,46]])
    color_bounds_upper = np.array([[77,255,255],[34,255,255],[10,255,255],[180,255,255],[25,255,255],[124,255,255],[155,255,255]])
    mask = np.zeros((image.shape[0],image.shape[1]),dtype=np.uint8)
    for i in  range(len(color_bounds_lower)):
          mask1 = cv2.inRange(hsv, color_bounds_lower[i], color_bounds_upper[i])
          mask = cv2.bitwise_or(mask1, mask)
    merge = cv2.bitwise_and(mask,binary)
    centers = []
    for pic,contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area>400 and area<4000):
            x,y,w,h = cv2.boundingRect(contour)
            print("Rect",x,y,w,h)
            pos,size,angle = cv2.minAreaRect(contour)
            center = [int(x+w/2),int(y+h/2)]
            centers.append(center)
            co = colors(hsv[center[1],center[0],:])
            cv2.putText(image,co,(x,y),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0))
            #pos = (int(pos[0]),int(pos[1]))
            #endpoint = (int(pos[0]+40*np.cos(angle)),int(pos[1]+40*np.sin(angle)))
            #image = cv2.line(image,pos,endpoint,(0,255,0),9)
            #image = cv2.rectangle(image, (int(pos[0]), int(pos[1])), (int(pos[0]) + int(size[0]), int(pos[1] + size[1])), (0, 0, 255), 2)
            image = cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
    center = np.array(centers)
    plt.imshow(image)
    plt.show()
    # grey_thresh = cv2.bitwise_and(grey_image,grey_image, mask=mask)
    # ret,thresh1 = cv2.threshold(grey_thresh,127,255,cv2.THRESH_BINARY)
    # thresh= cv2.bitwise_and(image,image, mask=mask)
if __name__ =="__main__":
    main()