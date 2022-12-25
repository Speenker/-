import scipy.special as sc
import matplotlib.pyplot as plt
from math import sin,cos,sqrt,log
import pandas as pd

df = pd.read_csv('climbfinal.csv', sep=';')

Altitude = df['AltitudeFromTerrain']
Time = df['Time']

#константы
G = 6.6743*10**(-11)
M = 5.972*10**24
R = 6378100
U = 2300
m1 = 540300 
m_without_fuel = 256220
m2 = 224380
F1 = 10579000
F2 = 3827000
e = 2.71828

def Ftyag(m, AltitudeFromTerrain):
  return G*((M*m) / (R+AltitudeFromTerrain)**2)

def velocity(v0, h, m, m0):
  return v0-log(m/m0)*(U+G*(M/(R+h)**2))

#Построение графика ускорения от времени
Acc = [0]*268
T = list(range(1, 269))
H = [0]*268

for t in range(113):
  Acc[t] = (1.25*(F1 - Ftyag(m1-(1060+1556)*t, H[t])) / m1)
  if t+1 < 114:
    H[t+1] = (Acc[t]*((t+1)**2))/2
#  print('ftyag =', Ftyag(m1, H[t]))
#  print('acceleration =', Acc[t])
#  print('height =', H[t])  

for t in range(113, 268):
  Acc[t] = (1.25*(F2 - Ftyag(m2-1060*t, H[t])) / m2)
  if t+1 < 268:
    H[t+1] = (Acc[t]*((t+1)**2))/2
#  print('ftyag =', Ftyag(m1, H[t]))
#  print('acceleration =', Acc[t])
#  print('height =', H[t])  

plt.plot(T, Acc)
plt.title("Ускорение", fontsize = 17)
plt.ylabel("Ускорение (м/с^2)")
plt.xlabel("Время (c)")
plt.grid(True)
plt.show()

#plt.plot(T, H)
#plt.title("Высота от времени", fontsize = 17)
#plt.ylabel("Высота (м)")
#plt.xlabel("Время (c)")
#plt.grid(True)
#plt.show()

V = []
hm = []
k = 1060
dM = 284080
i = 0
v = velocity(0,0,m1,m1)
h = 0
lastM = m1
while(268-i >= 0 and dM >= 0):
  #print(i, ":", 3.5*v, "", 0.9*h)
  V.append(3.5*v)
  dM -= k
  v0 = velocity(v, h, m_without_fuel+dM, m_without_fuel+dM+k)
  hm.append(0.9*h)
  h += (v+v0)*0.5
  v = v0
  i += 1
V = V[1:]
hm = hm[1:]

plt.plot(T, V)
plt.title("Скорость", fontsize = 17)
plt.ylabel("Скорость (м/с)")
plt.xlabel("Время (c)")
plt.grid(True)
plt.show()

plt.plot(T, hm)
plt.title("Высота", fontsize = 17)
plt.ylabel("Скорость (м/с)")
plt.xlabel("Время (c)")
plt.grid(True)
plt.show()

"""**Далее представлены графики, демонстрирующие зависимость некоторых величин, взятых из KSP, от времени на этапе взлета**"""

import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('climbfinal.csv', sep=";")
df

df.plot("Time", "Velocity", ylabel = "Velocity")
plt.grid(True)

df.plot("Time", "GForce", ylabel = "GForce")
plt.grid(True)

df.plot("Time", "Acceleration", ylabel = "Acceleration")
plt.title("Ускорение KSP")
plt.grid(True)

df.plot("Time", "Thrust", ylabel = "Thrust")
plt.grid(True)
#Первое "падение" тяги (с 4635.95 до 1989.02) обусловлено отсоединением твердотопливных ускорителей
#Второе "падение" (с 2000 до 0) обусловлено отсоединением первой ступени

df.plot("Time", "TWR", ylabel = "TWR")
plt.grid(True)

df.plot("Time", "Mass", ylabel = "Mass")
plt.grid(True)

df.plot("Time", "AltitudeFromTerrain", ylabel = "AltitudeFromTerrain")
plt.grid(True)
