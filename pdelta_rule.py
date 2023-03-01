import numpy as np

def computeError(caracteristica,Pesos):
    sumas = []
    for z in caracteristica:
        resps = np.zeros((Pesos.shape[0]))
        for i in range(Pesos.shape[0]):
            resps[i] = z@Pesos[i]
        sumas.append(np.sum(resps))
    return sumas

def squashingFunction(caracteristica,Pesos):
    sumas = computeError(caracteristica,Pesos)
    estimacion = []
    for i in range(caracteristica.shape[0]):
        if sumas[i]<=-1:
            r = -1.0
        elif -1 < sumas[i] and sumas[i] < 1:
            r = sumas[i]
        else:
            r = 1.0
        estimacion.append(r)
    return np.array(estimacion)

def pDeltaRule(dato,peso,o,obar,mu,eta,gamma,vepsilon):
    ppto = dato@peso
    paso=0
    if obar > o + vepsilon and ppto >= 0:
        paso = -1.0*eta*dato
    if obar < o - vepsilon and ppto < 0:
        paso = eta*dato
    if obar <= o + vepsilon and 0 <= ppto and ppto < gamma:
        paso = eta*mu*dato
    if obar >= o - vepsilon and -gamma < ppto and ppto<0:
        paso = -1.0*eta*mu*dato
    return paso

def learnig_step(caracteristica,alphav,o,mu,eta,gamma,vepsilon,id):
    for i in range(caracteristica.shape[0]):
        obar = squashingFunction(caracteristica, alphav)
        alphav[id] = alphav[id] + pDeltaRule(caracteristica[i],alphav[id],o[i],obar[i],mu,eta,gamma,vepsilon)
        alphav[id]  = alphav[id]/np.linalg.norm(alphav[id])
    return alphav[id]

