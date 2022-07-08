import myfun
import matplotlib.pyplot as plt
import numpy as np

# compute entropy of generic binary random variable as function of p0 and plot it
#  step p0 (ogni quanto cambio il valore di p0)
delta = 0.0001
intervalli = int(1 / delta)
p0 = np.linspace(0, 1, intervalli)

y = []
# per ogni valore di p0 = i calcolo l'entropia della variabile aleatoria binaria corrispondente
for i in p0:
    # primo e ultimo valore 0 (log(0) e log(1))
    if i == 0 or i == 1:
        y.append(0)
        continue
    # probability mass function vector
    p = (i, 1 - i)
    # call entropy function
    h = myfun.entropy(p)
    y.append(h)

# creo grafico
plt.plot(p0, y, label="Entropia")
plt.xlabel("P0 probability")
plt.ylabel("Entropy value")
plt.title("Entropy of a discrete r.v function of p0")
plt.legend()
plt.show()

joint_prob = np.array([[0.1, 0, 0],
              [0.2, 0.3, 0.2],
              [0, 0, 0.2]])
marg_prob_Y = [0.3, 0.3, 0.4]
conditional_entropy = myfun.conditional_entropy2(joint_prob)
print(conditional_entropy)