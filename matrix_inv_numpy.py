from numpy.linalg import inv
import numpy as np

N = 4096 
TIMES = 1 

def main():
    for i in range(TIMES):
        array = np.random.uniform(0,100,(N,N)) 
        a = np.array(array)
        res =inv(np.matrix(a)) 

        check = np.matmul(res, a)
        if(check != np.eye(N)).all():
            print("KO")

    print("OK")




if __name__ == "__main__":
    main()
