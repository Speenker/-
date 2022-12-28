import scipy.special as sc
import matplotlib.pyplot as plt
from math import sin,cos,sqrt,log
import pandas as pd

df = pd.read_csv('climbfinal.csv', sep=';')

Altitude = df['AltitudeFromTerrain']
Time = df['Time']

#константы из ксп
G = 6.6743*10**(-11)
g = 9.81
M = 5.291*10**22
R = 600000
U = 2300
m1 = 292650
m2 = 224049
F1 = 4248400
F2 = 749200

def Ftyag(m, AltitudeFromTerrain):
  return G*((M*m) / (R+AltitudeFromTerrain)**2)

def velocity(v0, h, m, m0, k, t):
#  return v0-log(m/m0)*(U+G*(M/(R+h)**2))
  return -(U*k*t)-Ftyag(m, h)

#Построение графика ускорения
Acc = [0]*179
T = list(range(1, 180))
H = [0]*179

for t in range(64):
  Acc[t] = ((F1 - Ftyag(m1-(4*1120+1818)*(t+1), Altitude[t])) / m1)
#  if t+1 < 65:
#    H[t+1] = (Acc[t]*((t+1)**2))/2

#Вывод промежуточных данных
#  print('ftyag =', Ftyag(m1, H[t]))
#  print('acceleration =', Acc[t])
#  print('height =', H[t])  

for t in range(64, 179):
  Acc[t] = ((F2 - Ftyag(m2-2281*2*(t+1), Altitude[t] )) / m2)
#  if t+1 < 188:
#    H[t+1] = (Acc[t]*((t+1)**2))/2

#Вывод промежуточных данных
#  print('ftyag =', Ftyag(m1, H[t]))
#  print('Acceleration =', Acc[t])
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

#Построения графика скорости
V = []
hm = []
fuel = 39000 + 64800
fuel1 = 39000 + 18624
fuel2 = fuel - fuel1 #46176
m1_without_fuel = m1 - fuel1
m2_without_fuel = m2 - fuel2
k1 = fuel1 / 64 #единица затраты топлива в секунду на первой ступени
k2 = fuel2 / 116 #единица затраты топлива в секунду на второй ступени
dM = fuel #запас топлива
t = 0
v = velocity(0,0,m1,m1,k1,0)
h = 0

while(64 - t >= 0 and dM >= 0): #рассчет скорости до отделения 1-ой ступени
#  print("Time:", t, "Velocity:", v, "Height:", h) #Вывод промежуточных данных
  V.append(v)
  dM -= k1 #уменьшение топлива на единицу затраты
  v0 = velocity(v, h, m1_without_fuel + dM, m1_without_fuel + dM + k1, k1, t)
  hm.append(h)
  h += (v + v0)*0.5
  v = v0
  t += 1

while(179 - t >= 0 and dM >= 0): #продолжение рассчета, работает 2-ая ступень
#  print("Time:", t, "Velocity:", v, "Height:", h) #Вывод промежуточных данных
  V.append(v)
  dM -= k2
  v0 = velocity(v, h, m2_without_fuel + dM, m2_without_fuel + dM + k2, k2, t)
  hm.append(h)
  h += (v + v0)*0.5
  v = v0
  t += 1

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

"""**Далее представлены графики, демонстрирующие зависимость некоторых величин, взятых из KSP, от времени на этапе взлета, а также сравнительные графики величин из KSP, с полученными 
путем симуляции**
"""

import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('climbfinal.csv', sep=";")
df

df.plot("Time", "Velocity", label = "KSP")
plt.plot(T, V, label = "Simulation")
plt.legend()
plt.title("Velocity", fontsize = 17)
plt.grid(True)

df.plot("Time", "Acceleration", label = "KSP")
plt.plot(T, Acc, label = "Simulation")
plt.legend()
plt.title("Acceleration", fontsize = 17)
plt.grid(True)

df.plot("Time", "AltitudeFromTerrain", label = "KSP")
plt.plot(T, hm, label = "Simulation")
plt.legend()
plt.title("AltitudeFromTerrain", fontsize = 17)
plt.grid(True)

df.plot("Time", "GForce", label = "GForce")
plt.title("GForce")
plt.grid(True)

df.plot("Time", "Thrust", label = "Thrust")
plt.title("Thrust")
plt.grid(True)
#Первое "падение" тяги (с 4635.95 до 1989.02) обусловлено отсоединением твердотопливных ускорителей
#Второе "падение" (с 2000 до 0) обусловлено отсоединением первой ступени

df.plot("Time", "TWR", label = "TWR")
plt.title("TWR")
plt.grid(True)

df.plot("Time", "Mass", label = "Mass")
plt.title("Mass")
plt.grid(True)