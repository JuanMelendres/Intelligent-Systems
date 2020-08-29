import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("Izquierda, derecha, cerrado.txt")
samp_rate = 256
samps = data.shape[0]
n_channel = data.shape[1]

print("Numero de muestras: ", data.shape[0])
print("Numero de canales: ", data.shape[1])
print("Duracion de registros: ", samps / samp_rate, "segundos")

#print(data)

# Time Channel
time = data[:, 0]

# Data Channels (extract data)
chann1 = data[:, 1]
chann2 = data[:, 3]

'''
#plt.plot(time, chann1, label = "Channel 1")
#plt.plot(time, chann2, color = 'red', label = "Channel 2")
#plt.xlabel("Tiempo (s)")
#plt.ylabel("microvolts")
#plt.legend()
#plt.show()

plt.plot(time[200:500], chann1[200:500], label = "Channel 1")
plt.plot(time[200:500], chann2[200:500], color = 'red', label = "Channel 2")
plt.xlabel("Tiempo (s)")
plt.ylabel("microvolts")
plt.legend()
plt.show()
'''
# MArks

mark = data[:, 6]
# Muestra 201 +
# Muestra 202 ->
# Muestra 203 preparacion

# Rango importante
# Muestra 101 condicion1
# Muestra 200 descanso

# Simple For
'''
for i in range(samps):
    if mark[i] > 0:
        print("Marca", mark[i], "Muestra", i, "Tiempo", time[i])
'''

trainig_samples = {}

for i in range(0, samps):
    if mark[i] > 0:
        print('Marca', mark[i], 'Muestra', i, 'Tiempo', time[i])

        if (mark[i] > 100) and (mark[i] < 200):
            initSamp = i
            condition_id = mark[i]
        elif mark[i] == 200:
            if not condition_id in trainig_samples.keys():
                trainig_samples[condition_id] = []
            trainig_samples[int(condition_id)].append([initSamp, i])

print("Rango de muestras con datos de entrenamiento: ", trainig_samples)
'''
start_samp = trainig_samples[102][2][0];
end_samp = trainig_samples[102][2][1];
plt.plot(time[start_samp:end_samp], chann1[start_samp:end_samp], label = "Channel 1")
plt.plot(time[start_samp:end_samp], chann2[start_samp:end_samp], color = "red", label = "Channel 2")
plt.xlabel("Tiempo (s)")
plt.ylabel("microvolts")
plt.legend()
plt.show()
'''
# Power spectral density (PSD)
win_size = 256
init_samp = trainig_samples[103][2][0]
end_samp = init_samp + win_size

x = chann1[init_samp : end_samp]
t = time[init_samp : end_samp]

power, freq = plt.psd(x, NFFT = win_size, Fs = samp_rate)
#plt.show()

start_freq = next(x for x, val in enumerate(freq) if val >= 4.0)
end_freq = next(x for x, val in enumerate(freq) if val >= 60.0)
print(start_freq, end_freq)

start_index = np.where(freq >= 4.0)[0][0]
end_index = np.where(freq >= 60.0)[0][0]

plt.plot(freq[start_index:end_index], power[start_index:end_index])
plt.xlabel("Hz")
plt.ylabel("Power")
plt.show()