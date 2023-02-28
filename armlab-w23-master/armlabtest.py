import numpy as np 

def main():

    k = np.array([[952.45,0,650.0677],[0,963.94,344.82],[0,0,1]])
    h = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,990],[0,0,0,1]])
    k_inv = np.linalg.inv(k)
    print(k_inv)
    came_frame = 990*np.matmul(k_inv,np.array([600,400,1]))
    came_frame = np.array([came_frame[0],came_frame[1],came_frame[2],1]).T
    world_frame = np.matmul(h,came_frame)
    print(world_frame)

main()