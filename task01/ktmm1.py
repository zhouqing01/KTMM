from tkinter import *
import csv
from scipy.optimize import fsolve
from scipy.integrate import odeint
import numpy as np
from OpenGL.GL import *
import sys
import pygame
from OpenGL.GLU import *
from pygame.locals import *
import sympy as sm
from sympy import solve_poly_system
import matplotlib.pyplot as plt

# Python Tkinter 文本框用来让用户输入一行文本字符串
def check():
    s1 = str(e1.get()) #获取文本框内容
    s2 = str(e2.get())
    s3 = str(e3.get())
    global name_file
    global name_file_s_parameter
    global name_file_print
    name_file = s1
    name_file_s_parameter = s2
    name_file_print = s3
    root.destroy() #destroy()只是终止mainloop并删除所有小部件


root = Tk()
root.geometry("400x200")
root.title("Настройки")
root["bg"] = 'green'
e1 = Entry() #文本框
e2 = Entry()
e3 = Entry()
e1.place(x=200, y=10) #文本框位置
e2.place(x=200, y=50)
e3.place(x=200, y=90)
Button(root, text='Передать программе на обработку.', bg='yellow', command=check).place(x=90, y=130) #Tkinter 按钮组件
text1 = Label(root, text="Введите файл с моделью:", bg='cyan')
text1.place(x=10, y=10)
text2 = Label(root, text="Введите файл с параметрами:", bg='cyan')
text2.place(x=10, y=50)
text3 = Label(root, text="Выходный файл:", bg='cyan')
text3.place(x=10, y=90)
root.mainloop() #检测窗口相应的事件，=while true
name_file = name_file.replace('\\', '/')
name_file_s_parameter = name_file_s_parameter.replace('\\', '/')
# massiv_parametrov = parameter(name_file_s_parameter)

with open(name_file_s_parameter, newline='') as f:
    line_array = []
    reader = csv.reader(f)
    for row in reader:
        lines = [x for x in ';'.join(row).split(';')]
        line_array.append(lines)

massiv_parametrov = np.array(list(map(float, line_array[1])))


class objloader:
    def __init__(self): #
        self.vert_coords = []
        self.vertex_index = []
        self.part = 0

    def load_model(self, file): #引入模型文件，对文件进行操作
        schetchik_strok = 0
        massiv_index_strok = []
        for line in open(file, 'r'):
            schetchik_strok = schetchik_strok + 1
            values = line.split()
            if (len(values) > 1) and (values[1] == 'object'):
                self.part = self.part + 1
                massiv_index_strok.append(schetchik_strok)
        part = []
        part_index = [[] for i in range(self.part)]
        k = 0
        w = 0
        for line in open(file, 'r'):
            values = line.split()
            k = k + 1
            if len(values) > 1 and values[0] == 'v':
                part.append(tuple([float(t) for t in values[1:4]]))
            if ((massiv_index_strok[w] < k) and (k < massiv_index_strok[w + 1]) and (w < self.part - 1)):
                if len(values) > 1 and values[0] == 'f':
                    part_index[w].append(tuple([int(t) - 1 for t in values[1:4]]))
            if k == massiv_index_strok[w + 1] and w + 1 < self.part - 1:
                w = w + 1
                continue
            if k > massiv_index_strok[self.part - 1]:
                if len(values) > 1 and values[0] == 'f':
                    part_index[self.part - 1].append(tuple([int(t) - 1 for t in values[1:4]]))
        self.vert_coords = tuple(part) #‘v’
        self.vertex_index = tuple([tuple(part_index[i]) for i in range(self.part)]) #‘f’


obj = objloader()
obj.load_model(name_file) #对obj文件进行读取操作
vertices = obj.vert_coords #‘v’顶点坐标
edges = obj.vertex_index #‘f‘为边缘
r = obj.part

level = [0, 6, 8, 9]

def square(m, s, q):
    a = [s[i] - m[i] for i in range(3)]
    b = [q[i] - m[i] for i in range(3)]
    return 0.5 * np.linalg.norm(np.array([(a[1] * b[2] - a[2] * b[1]), (a[2] * b[0] - a[0] * b[2]), (a[0] * b[1] - a[1] * b[0])]))


square_search = []
for k in range(r - 1):
    square_section = 0
    square_section1 = 0
    for edge in edges[k]:
        if (vertices[edge[0]][1] == level[k]) and (vertices[edge[1]][1] == level[k]) and (vertices[edge[2]][1] == level[k]):
            square_section = square_section + square(vertices[edge[0]], vertices[edge[1]], vertices[edge[2]])
    for edge in edges[k + 1]:
        if (vertices[edge[0]][1] == level[k]) and (vertices[edge[1]][1] == level[k]) and (vertices[edge[2]][1] == level[k]):
            square_section1 = square_section1 + square(vertices[edge[0]], vertices[edge[1]], vertices[edge[2]])
    square_search.append(min(square_section, square_section1)) # площадь сечения, разделяющего i-ый и j-ый КЭ
square_area_element = []
for k in range(r):
    square_pov = 0
    for edge in edges[k]:
        square_pov = square_pov + square(vertices[edge[0]], vertices[edge[1]], vertices[edge[2]])
    square_area_element.append(square_pov)

epsilon = np.array(list(map(float, line_array[0])))
c = np.array(list(map(float, line_array[1])))
lambdi = np.zeros((5, 5))
lambdi[0][1] = float(line_array[3][0])
lambdi[1][2] = float(line_array[4][0])
lambdi[2][3] = float(line_array[5][0])
lambdi[3][4] = float(line_array[6][0])
A = massiv_parametrov[0]

Q_R_0 = float(line_array[2][0])

def Q_R_1(t):
    return (A * (20 + 3 * np.sin(t / 4)))

Q_R_2 = float(line_array[2][2])
Q_R_3 = float(line_array[2][3])
Q_R_4 = float(line_array[2][4])
K = np.zeros((5, 5))
K[0][1] = lambdi[0][1] * square_search[0]
K[1][2] = lambdi[1][2] * square_search[1]
K[2][3] = lambdi[2][3] * square_search[2]
K[3][4] = lambdi[3][4] * square_search[3]
C_0 = 5.67


def find_solution(z):
    T0 = z[0]
    T1 = z[1]
    T2 = z[2]
    T3 = z[3]
    T4 = z[4]
    f = np.empty(5)
    f[0] = -K[0][1] * (T1 - T0) - epsilon[0] * square_area_element[0] * C_0 * ((T0 / 100) ** 4)
    f[1] = -K[0][1] * (T1 - T0) - K[1][2] * (T2 - T1) - epsilon[1] * square_area_element[1] * C_0 * ((T1 / 100) ** 4) + 20 * A
    f[2] = -K[1][2] * (T2 - T1) - K[2][3] * (T3 - T2) - epsilon[2] * square_area_element[2] * C_0 * ((T2 / 100) ** 4)
    f[3] = -K[2][3] * (T3 - T2) - K[3][4] * (T4 - T3) - epsilon[3] * square_area_element[3] * C_0 * ((T3 / 100) ** 4)
    f[4] = -K[3][4] * (T4 - T3) - epsilon[4] * square_area_element[4] * C_0 * ((T4 / 100) ** 4)
    return (f)

start = np.array(massiv_parametrov[0:6])
p = fsolve(find_solution, start)
print(start)
print('стационарные решения=', p) # стационарные решения


def g(y, x):
    T0 = y[0]
    T1 = y[1]
    T2 = y[2]
    T3 = y[3]
    T4 = y[4]
    f0 = (-K[0][1] * (T1 - T0) - epsilon[0] * square_area_element[0] * C_0 * ((T0 / 100) ** 4)) / c[0]
    f1 = (-K[0][1] * (T1 - T0) - K[1][2] * (T2 - T1) - epsilon[1] * square_area_element[1] * C_0 * ((T1 / 100) ** 4) + 20 * A) / c[1]
    f2 = (-K[1][2] * (T2 - T1) - K[2][3] * (T3 - T2) - epsilon[2] * square_area_element[2] * C_0 * ((T2 / 100) ** 4)) / c[2]
    f3 = (-K[2][3] * (T3 - T2) - K[3][4] * (T4 - T3) - epsilon[3] * square_area_element[3] * C_0 * ((T3 / 100) ** 4)) / c[3]
    f4 = (-K[3][4] * (T4 - T3) - epsilon[4] * square_area_element[4] * C_0 * ((T4 / 100) ** 4)) / c[4]
    return [f0, f1, f2, f3, f4]


# t = np.linspace(0, 15000, 150000)
t = np.linspace(0, 12, 200)

sol = odeint(g, start, t)
# sol = odeint(g, p, t)
plt.figure(1)
plt.grid(True)
plt.xlabel('t')
plt.ylabel('Temperature')
plt.plot(t, sol[:, 0], color='g', label='P1')
plt.plot(t, sol[:, 1], color='r', label='P2')
plt.plot(t, sol[:, 2], color='b', label='P3')
plt.plot(t, sol[:, 3], color='purple', label='P4')
plt.plot(t, sol[:, 4], color='black', label='P5')
plt.legend()
plt.show()
colors = ((1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 1), (0.7, 0.2, 1), (1, 0.5, 0.5))

headers = ('t', 'Part1', 'Part2', 'Part3', 'Part4', 'Part5')
with open(name_file_print, 'w') as f:
    f_csv = csv.DictWriter(f, fieldnames=['t', 'Part1', 'Part2', 'Part3', 'Part4', 'Part5'])
    f_csv.writerow({'t': 't', 'Part1': 'Part1', 'Part2': 'Part2', 'Part3': 'Part3', 'Part4': 'Part4', 'Part5': 'Part5'})
    for i in range(len(t)):
        f_csv.writerow({'t': t[i], 'Part1': sol[i, 0], 'Part2': sol[i, 1], 'Part3': sol[i, 2], 'Part4': sol[i, 3], 'Part5': sol[i, 4]})


def model(): #
    glBegin(GL_TRIANGLES)
    for g in range(r):
        for edge in edges[g]:
            for vertex in edge:
                glColor3fv(colors[g]) #设定不同区域的颜色
                glVertex3fv(vertices[vertex]) #引入空白的模型
    glEnd()


def main():
    pygame.init()
    display = (600, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    gluPerspective(42, (display[0] / display[1]), 0.1, 50)
    glTranslatef(0, 0, -40) #z轴平移，相当于缩小模型
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        glRotatef(10, 1, 1, 1) #绕原点逆时针旋转10度
        glClearColor(0.1, 0.8, 0.1, 0) #分别为红绿蓝和alpha值
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT) #清楚缓冲区
        model()  #引入模型
        pygame.display.flip() #更新屏幕显示
        pygame.time.wait(10)

main()