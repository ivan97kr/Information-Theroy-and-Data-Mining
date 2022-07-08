import os
import matplotlib.pyplot as plt
import numpy as np
import pygit2
import pandas    as pd

# info repo to clone
repoURL_ita = 'https://github.com/pcm-dpc/COVID-19.git'
repoURL_global = 'https://github.com/CSSEGISandData/COVID-19'
local_dir_ita = './PythonCOVID'
local_dir_global = './PythonCOVID_global'

# solo se non esiste già la scarico completamente
if not os.path.isdir(local_dir_ita):
    repoClone = pygit2.clone_repository(repoURL_ita, local_dir_ita)
# altrimenti only update
else:
    os.system('cd PythonCOVID ; pwd ; git fetch ; git pull')  # pwd to verify working dir

# stessa cosa per la repo globale
if not os.path.isdir(local_dir_global):
    repoClone = pygit2.clone_repository(repoURL_global, local_dir_global)
# else only update
else:
    os.system('cd PythonCOVID_global ; pwd ; git fetch ; git pull')  # pwd to verify working dir

# csv di interesse per stat nazionali (uso pandas lib)
file = 'PythonCOVID/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv'
# leggo solo colonne di interesse
# colonna 1 positivi, colonna 2 deceduti
data = pd.read_csv(file, usecols=['nuovi_positivi', 'deceduti'])  # leggo file csv con pandas lib
nuovi_positivi = data.take([0], axis=1)  # colonna 1
deceduti = data.take([1], axis=1)   # colonna 2

# grafico andamento contagi nazionale
x1 = np.linspace(0, len(nuovi_positivi) - 1, len(nuovi_positivi))  # asse x, giorni
plt.plot(x1, nuovi_positivi, label="Nuovi contagi")
plt.xlabel("Giorni")
plt.ylabel("Contagi Giornalieri")
plt.title("Nuovi contagi giornalieri nazionali dal 24/02 ad oggi")
plt.legend()
plt.show()

# grafico andamento deceduti
x2 = np.linspace(0, len(deceduti) - 1, len(deceduti))  # asse x, giorni
plt.plot(x2, deceduti, label="Decessi")
plt.xlabel("Giorni")
plt.ylabel("Decessi Giornalieri")
plt.title("Decessi giornalieri nazionali dal 24/02 ad oggi")
plt.legend()
plt.show()

# dir csv di interesse dati globali
file = 'PythonCOVID_global/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
data = pd.read_csv(file)  # leggo file csv con pandas lib
# elimino colonne non utili (lat e long)
final_data = data.drop(['Lat', 'Long'], axis=1)

# sommo tutti i valori di ogni colonna
# (ogni colonna è un giorno e ogni riga è uno stato)
positive_cases_global = final_data.sum(axis=0, skipna=True, numeric_only=True)

# grafico contagi mondiali
x3 = np.linspace(0, len(positive_cases_global) - 1, len(positive_cases_global))  # asse x, giorni
plt.plot(x3, positive_cases_global, label="Global positive")
plt.xlabel("Giorni")
plt.ylabel("Positivi giornalieri globali")
plt.title("Positivi giornalieri globali dal 22/02 ad oggi")
plt.legend()
plt.show()
