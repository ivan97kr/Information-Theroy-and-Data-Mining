import math


#   entropy fun:
#   computes the entropy of a discrete random variable given its probability mass function p = [p1; p2; :::; pN].
def entropy(p):
    #  lunghezza del vettore massa di probabilità
    n = len(p)
    #   var per il calcolo dell'entropia
    h = 0
    #  sommatoria che costituisce la entropy function
    for i in range(0, n):
        h = h + p[i] * math.log((1 / p[i]), 2)
    return h


#   joint_entropy fun:
#   computes the joint entropy of two generic discrete random variables given their joint p.m.f.
#   input example:
#   pxy = np.array([[1/12, 1/12, 1/12], [1/12, 1/12, 1/12], [1/12, 1/12, 1/12], [1/12, 1/12, 1/12]])
def joint_entropy(pxy):
    #  numero delle righe (possibili eventi v.a. x)
    nx = pxy.shape[0]  # 0 conta le righe, 1 conta le colonne
    #  numero delle colonne (possibili eventi v.a. y)
    ny = pxy.shape[1]
    # var per il calcolo dell'entropia
    h = 0
    #  doppia sommatoria che costituisce la joint entropy function
    for i in range(0, nx):
        for j in range(0, ny):
            h = h + pxy[i][j] * math.log((1 / pxy[i][j]), 2)
    return h


#   conditional entropy fun:
#   computes the conditional entropy of two generic discrete random variables given their joint and marginal p.m    .f
def conditional_entropy1(px, py, pxy):
    #  numero delle righe (possibili eventi v.a. x)
    nx = len(px)
    #  numero delle colonne (possibili eventi v.a. y)
    ny = len(py)
    # var per il calcolo dell'entropia
    h = 0
    #  doppia sommatoria che costituisce la conditional entropy function
    for i in range(0, nx):
        for j in range(0, ny):
            h = h + (pxy[i][j] * math.log((py[j] / (pxy[i][j])), 2))
    return h


#   conditional entropy fun:
#   come input solo la p. congiunta dalla quale è possibile ricavare le marginali sommando rispettivamente le righe
#   e le colonne
def conditional_entropy2(pxy):
    #  numero delle righe (possibili eventi v.a. x)
    nx = pxy.shape[0]
    #  numero delle colonne (possibili eventi v.a. y)
    ny = pxy.shape[1]
    #   calcolo la probabilità marginale di y
    py = pxy.sum(axis=0)  # somma tutti gli elementi di ogni riga, restituisce vettore colonna
    # var per il calcolo dell'entropia
    h = 0
    #  doppia sommatoria che costituisce la conditional entropy function
    for i in range(0, nx):
        for j in range(0, ny):
            if py[j] == 0 or pxy[i][j] == 0:
                continue
            h = h + (pxy[i][j] * math.log((py[j] / (pxy[i][j])), 2))
    return h


#   mutual information fun:
#   computes the mutual information of two generic discrete random variables given their joint and marginal p.m.f.
def mutual_information1(pxy, px, py):
    #  numero delle righe (possibili eventi v.a. x)
    nx = len(px)
    #  numero delle colonne (possibili eventi v.a. y)
    ny = len(py)
    # var per il calcolo dell'entropia
    h = 0
    #  doppia sommatoria che costituisce la conditional entropy function
    for i in range(0, nx):
        for j in range(0, ny):
            h = h + (pxy[i][j] * math.log(((pxy[i][j]) / (px[i] * py[j])), 2))
    return h


#   computes the mutual information of two generic discrete random variables given only their joint p.m.f.
def mutual_information2(pxy):
    #  numero delle righe (possibili eventi v.a. x)
    nx = pxy.shape[0]
    #  numero delle colonne (possibili eventi v.a. y)
    ny = pxy.shape[1]
    #   calcolo la probabilità marginale di x
    py = pxy.sum(axis=0)  # 0 somma tutti gli elementi di ogni colonna, restituisce vettore riga
    #   calcolo la probabilità marginale di y
    px = pxy.sum(axis=1)  # 1 somma tutti gli elementi di ogni riga, restituisce vettore colonna
    # var per il calcolo dell'entropia
    h = 0
    #  doppia sommatoria che costituisce la conditional entropy function
    for i in range(0, nx):
        for j in range(0, ny):
            h = h + (pxy[i][j] * math.log(((pxy[i][j]) / (px[i] * py[j])), 2))
    return h


#   normalized conditional entropy fun
#   input prob marginali e congiunta
def normalized_conditional_entropy(px, py, pxy):
    h = conditional_entropy1(px, py, pxy) / entropy(px)
    return h


#   normalized joint entropy fun
#   input prob congiunta
def normalized_joint_entropy(pxy):
    px = pxy.sum(axis=0)
    py = pxy.sum(axis=1)
    h = joint_entropy(pxy) / (entropy(px) + entropy(py))
    return h


#   normalized mutual information type 1 fun
#   input prob congiunta
def normalized_mutual_information1(pxy):
    h = (1 / normalized_joint_entropy(pxy)) - 1
    return h


#   normalized mutual information type 2 fun
#   input prob congiunta
def normalized_mutual_information2(pxy):
    h = 1 + normalized_mutual_information1(pxy)
    return h


#   normalized mutual information type 3 fun
#   input prob marginali e congiunta
def normalized_mutual_information3(px, py, pxy):
    h = (mutual_information2(pxy) / (math.sqrt(entropy(px) * entropy(py))))
    return h
