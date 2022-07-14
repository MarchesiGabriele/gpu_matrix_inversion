from numpy.linalg import inv
import numpy as np
import time

N = 8192 
TIMES = 1 

def main():
    for i in range(TIMES):
        start = time.monotonic()
        array = np.random.uniform(0,100,(N,N)) 
        a = np.array(array)
        res =inv(np.matrix(a)) 
        end = time.monotonic()
        check = np.matmul(res, a)
        if(check != np.eye(N)).all():
            print("KO")

    print(f"OK, tempo impiegato: {end-start}s")
    print(f"DIMENSIONE MATRICE: {N}")




if __name__ == "__main__":
    main()
