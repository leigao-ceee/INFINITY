import numpy as np
import matplotlib.pyplot as plt

gluc_data = np.hsplit(np.loadtxt("glucose_gen.dat"), [1])
meal_data = np.hsplit(np.loadtxt("meal_gen.dat"), [4])
meal_t = meal_data[0]
meal_q = meal_data[1]
t = gluc_data[0]
y = gluc_data[1]

k=1 / 120
dt = t - meal_t
IG = np.sum(
    0.5 * meal_q * k * np.exp(-k * dt) * (np.sign(dt) + 1),
    axis=1,
    keepdims=True,
)

plt.subplot(221)
plt.ylim(0,800)
plt.ylabel('Ii(uU/ml)')
plt.plot(t.squeeze(),y.transpose(1,0)[1],'b')
plt.subplot(222)
plt.ylim(80,240)
plt.ylabel('G(mg/dl)')
plt.plot(t.squeeze(),y.transpose(1,0)[2]/100,'b')
plt.subplot(223)
plt.ylim(0,700)
plt.ylabel('Ip(uU/ml)')
plt.plot(t.squeeze(),y.transpose(1,0)[3],'b')
plt.subplot(224)
plt.ylabel('IG(mg/min)')
plt.plot(t.squeeze(),IG.squeeze(),'b')


gluc_data = np.hsplit(np.loadtxt("glucose_pre.dat"), [1])
meal_data = np.hsplit(np.loadtxt("meal_pre.dat"), [4])
meal_t = meal_data[0]
meal_q = meal_data[1]
t = gluc_data[0]
y = gluc_data[1]

k = np.loadtxt('Results.dat')[4]
print('k:',k)

dt = t - meal_t
IG = np.sum(
    0.5 * meal_q * k * np.exp(-k * dt) * (np.sign(dt) + 1),
    axis=1,
    keepdims=True,
)

plt.subplot(221)
plt.ylim(0,800)
plt.ylabel('Ii(uU/ml)')
plt.plot(t.squeeze(),y.transpose(1,0)[1],'r--')
plt.subplot(222)
plt.ylim(80,240)
plt.ylabel('G(mg/dl)')
plt.plot(t.squeeze(),y.transpose(1,0)[2]/100,'r--')
plt.subplot(223)
plt.ylim(0,700)
plt.ylabel('Ip(uU/ml)')
plt.plot(t.squeeze(),y.transpose(1,0)[3],'r--')
plt.subplot(224)
plt.ylabel('IG(mg/min)')
plt.plot(t.squeeze(),IG.squeeze(),'r--')

plt.show()