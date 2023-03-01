import numpy as np
from mpi4py import MPI
import pandas as pd
import random

from pdelta_rule import learnig_step, squashingFunction

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def trainingPerceptron(Datos,clases, pesos, nprocesos,gamma,mu,vepsilon,epocas):
    for i in range(epocas):
        eta = 1.0/(4*np.sqrt(i+1))
        if rank == 0:
            print("Epoca: ", i, "Valor de eta: ", eta)
            print("pesos: ",pesos)
        pesos = comm.bcast(pesos, root=0)
        peso = learnig_step(Datos, pesos, clases,mu, eta, gamma, vepsilon, rank)
        buf = None
        if rank == 0:
            buf = np.empty([nprocesos, 5], dtype='f')
        comm.Gather(peso,buf,root=0)
        if rank == 0:
            pesos = np.copy(buf)
    return pesos

def mezclaDatos(datos, clases):
    concatena = list(zip(datos,clases))
    random.shuffle(concatena)
    datos, clases = zip(*concatena)
    return np.array(datos), np.array(clases)

if __name__ == '__main__':

				parametros = None
				if rank == 0:
								File = input("Nombre del Archivo CSV: ")
								nprocesos = size
								gamma = float(input("Valor de Gamma: "))
								mu = float(input("Valor de mu: "))
								vepsilon = float(input("Valor de epsilon: "))
								epocas = int(input("Valor de Epocas: "))
								parametros = [File,nprocesos,gamma,mu,vepsilon,epocas]
				parametros = comm.bcast(parametros,root = 0)

				File = parametros[0]
				nprocesos = parametros[1]
				gamma = parametros[2]
				mu = parametros[3]
				vepsilon = parametros[4]
				epocas = parametros[5]

				clases = ['Iris-versicolor','Iris-virginica']

				data = pd.read_csv(File)
				data.drop('Id',axis=1,inplace=True)
				data1 = data[data.Species==clases[0]].values
				data2 = data[data.Species==clases[1]].values
				class1 = np.ones((data1.shape[0]))
				class2 = np.ones((data2.shape[0]))*(-1.0)
				data1[:,4] = np.ones((data1.shape[0]))
				data2[:,4] = np.ones((data2.shape[0]))

				datosVV25 = np.array(np.concatenate((data1[0:25],data2[0:25])))
				datosVV50 = np.array(np.concatenate((data1[25:50],data2[25:50])))
				claseVV25 = np.array(np.concatenate((class1[0:25],class2[0:25])))
				claseVV50 = np.array(np.concatenate((class1[25:50],class2[25:50])))
				datosVV25, claseVV25 = mezclaDatos(datosVV25,claseVV25)
				for i in range(50):
								datosVV25[i] = datosVV25[i]/np.linalg.norm(datosVV25[i])
								datosVV50[i] = datosVV50[i]/np.linalg.norm(datosVV50[i])

				pesos = np.empty((nprocesos,datosVV25.shape[1]),dtype='f')
				for i in range(nprocesos):
									peso = np.random.uniform(0,1,(5))
									pesos[i] =peso/np.linalg.norm(peso)

				we = trainingPerceptron(datosVV25,claseVV25,pesos,nprocesos,gamma,mu,vepsilon,epocas)

				if rank == 0:
								etbar = squashingFunction(datosVV25,we) 
								i=0
								for j in range(len(claseVV25)):
												if etbar[j] >= 0:
																tmp = 1
												else:
																tmp = -1
												if claseVV25[j] == tmp:
																i += 1 
								tstrain = float((i*100)/len(etbar))    
								etiquetaNW = squashingFunction(datosVV50,we)
								i=0
								for j in range(len(claseVV50)):
												if etiquetaNW[j] >= 0:
																tmp = 1
												else:
																tmp = -1
												if claseVV50[j] == tmp:
																i += 1 
								pstrain = float((i*100)/len(etiquetaNW))  
								print("Porcentaje de clasificación correcta Train =",tstrain)
								print("Porcentaje de clasificación correcta Test =",pstrain)
