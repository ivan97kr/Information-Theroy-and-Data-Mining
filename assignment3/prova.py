import numpy as np
import matplotlib.pyplot as plt
import myfun

# construction of a probability density function to se the trend of it:

# resolution of my function
# for simplicity consider a r.v. with uniform distribution with parameter a = 1/4 and b = 1/2
delta = 0.0001
interval = int(1 / delta)
a = 0.25
b = 0.5
x = np.linspace(0, 10, interval)

# built the domain, let's construct the codomain
y = []
# prob. density function is 0 out the (a,b) interval and 1/b-a in this interval
for i in x:
    if i < a or i > b:
        y.append(0)
    else:
        y.append(1 / (b - a))

#  plot to see later if differential entropy fun compute correct values
plt.plot(x, y, label="Probability density function")
plt.xlabel("x value")
plt.ylabel("f0 value")
plt.title("Uniform density function")
plt.legend()
plt.show()


