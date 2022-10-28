import numpy as np
import time

def mat_inv():
    file = open("C:/TESI/TESI DOCUMENTAZIONE/BENCHMARKS/NUMPY_64.txt", "w")
        
    i = 10
    while i < 5000:
        array = np.random.uniform(0, 100, (i,i)) 
        a = np.array(array)

        #HOLLOW
        for x in range(len(a)):
            a[x][x] = 0

        start = time.monotonic()
        res = inv(np.matrix(a)) 
        end = time.monotonic()

        check = np.matmul(res, a)

        #calcolo norma frobenius
        arr = np.squeeze(check)
        vec = arr @ arr

        somma = np.sum(vec) 

        print(f"Somma: {somma}, errore: {np.sqrt(i)-np.sqrt(somma)}")
        file.write(f"{i} {end-start} {np.sqrt(i)-np.sqrt(somma)}\n") 

        if i < 2000:
            i+= 10
        else:
            i += 1000

    file.close()


def just_inv(K):
    array = np.random.uniform(0, 100, (K,K)) 
    a = np.array(array)

    start = time.monotonic()
    res = np.linalg.inv(np.matrix(a)) 
    end = time.monotonic()
    print(f"TIME: {end-start}")


if __name__ == "__main__":
    #mat_inv()
    just_inv(10000)

