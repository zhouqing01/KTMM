import numpy as np
import math

import OpenGL
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5 import QtOpenGL

import OpenGL.GL as gl
from OpenGL import GLU

from scipy.integrate import odeint

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PyQt5.QtCore import QThread
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import Process, Queue

import threading
from multiprocessing import Process

from ktmm2_2 import main

import sys
import time

import cythv

import pyopencl as cl

# cntxt = cl.create_some_context()




class Verlet_Thread(QThread):
    def __init__(self, mainGLW, pl_i, parent=None):
        super().__init__()
        self.mainGLW = mainGLW
        self.pl_i = pl_i

    def run(self):
        for j in range(len(self.mainGLW.pos_x)):
            if j != self.pl_i and self.pl_i != 0:
                self.mainGLW.a[self.pl_i][0] += self.mainGLW.G * self.mainGLW.mas[j] * (
                            self.mainGLW.pos_x[j] - self.mainGLW.pos_x[self.pl_i]) / \
                                                (math.sqrt(
                                                    (self.mainGLW.pos_x[j] - self.mainGLW.pos_x[self.pl_i]) ** 2 + (
                                                                self.mainGLW.pos_y[j] - self.mainGLW.pos_y[
                                                            self.pl_i]) ** 2)) ** 3

                self.mainGLW.a[self.pl_i][1] += self.mainGLW.G * self.mainGLW.mas[j] * (
                            self.mainGLW.pos_y[j] - self.mainGLW.pos_y[self.pl_i]) / \
                                                (math.sqrt(
                                                    (self.mainGLW.pos_x[j] - self.mainGLW.pos_x[self.pl_i]) ** 2 + (
                                                                self.mainGLW.pos_y[j] - self.mainGLW.pos_y[
                                                            self.pl_i]) ** 2)) ** 3



platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
cntxt = cl.Context([device])
queue = cl.CommandQueue(cntxt)
queue_speed = cl.CommandQueue(cntxt)


class GLWidget(QtOpenGL.QGLWidget):
    def __init__(self, parent=None):
        self.parent = parent
        QtOpenGL.QGLWidget.__init__(self, parent)

    def initializeGL(self):
        self.qglClearColor(QtGui.QColor(255, 255, 255))
        # self.qglClearColor(QtGui.QColor(0, 0, 0))
        gl.glEnable(gl.GL_DEPTH_TEST)

        self.scale = 1.0

        # self.radio = [1.0, 0.244, 0.6052, 0.6371, 0.339, 6.9911, 5.8232, 2.5362, 2.4622]
        self.radio = [1.0, 2.244, 2.6052, 2.6371, 2.339, 6.9911, 5.8232, 2.5362, 2.4622]
        self.color = [[1.0, 0.6, 0.0], [1.0, 0.8, 0.4], [1.0, 0.8, 0.0], [0.0, 0.0, 1.0], [0.8, 0.2, 0.2],
                      [1.0, 0.6, 0.2], [1.0, 0.8, 0.6], [0.6, 0.8, 1.0], [0.0, 0.4, 0.8]]

        self.G = 6.67 / 10 ** 11

        self.mas = [1989000, 0.32868, 4.81068, 5.97600, 0.63345, 1876.64328, 561.80376, 86.05440, 101.59200]
        self.mas = [i * 10 ** 4 for i in self.mas]

        self.num_elem = 9


        self.initialize()

        # self.initialize()

    def init_auto(self):
        self.t = int(self.parent.Time_param.text())
        #####
        # for t in range(int(self.parent.Time_param.text())):
        #     self.t = t
        #####

        self.delt_t = float(self.parent.Step_param.text())
        self.time = np.linspace(0, self.t, int(self.t / self.delt_t))

        # уменьшение в 10**10
        self.pos_x = [0, 5.7910006, 10.8199995, 14.9599951, 22.793992, 77.8330257, 142.9400028, 287.0989228,
                      450.4299579]
        self.pos_y = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.v_x = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.v_y = [0, 47.87, 35.02, 29.78, 24.13, 13.07, 9.69, 6.81, 5.43]
        self.v_y = [i / 10 ** 2 for i in self.v_y]

        self.num_elem = len(self.pos_x)

        self.a = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
        for i in range(self.num_elem):
            for j in range(self.num_elem):
                if j != i and i != 0:
                    self.a[i][0] += self.G * self.mas[j] * (self.pos_x[j] - self.pos_x[i]) / (
                        math.sqrt((self.pos_x[j] - self.pos_x[i]) ** 2 + (self.pos_y[j] - self.pos_y[i]) ** 2)) ** 3

                    self.a[i][1] += self.G * self.mas[j] * (self.pos_y[j] - self.pos_y[i]) / (
                        math.sqrt((self.pos_x[j] - self.pos_x[i]) ** 2 + (self.pos_y[j] - self.pos_y[i]) ** 2)) ** 3


    def initialize(self):
        self.t = int(self.parent.Time_param.text())
        #####
        # for t in range(int(self.parent.Time_param.text())):
        #     self.t = t
        #####

        self.delt_t = float(self.parent.Step_param.text())
        self.time = np.linspace(0, self.t, int(self.t / self.delt_t))

        # уменьшение в 10**10
        self.pos_x = [0, 5.7910006, 10.8199995, 14.9599951, 22.793992, 77.8330257, 142.9400028, 287.0989228, 450.4299579]
        self.pos_y = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.v_x = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.v_y = [0, 47.87, 35.02, 29.78, 24.13, 13.07, 9.69, 6.81, 5.43]
        self.v_y = [i / 10 ** 2 for i in self.v_y]

        self.num_elem = len(self.pos_x)

        self.a = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
        for i in range(self.num_elem):
            for j in range(self.num_elem):
                if j != i and i != 0:
                    self.a[i][0] += self.G * self.mas[j] * (self.pos_x[j] - self.pos_x[i]) / (math.sqrt((self.pos_x[j] - self.pos_x[i]) ** 2 + (self.pos_y[j] - self.pos_y[i]) ** 2)) ** 3

                    self.a[i][1] += self.G * self.mas[j] * (self.pos_y[j] - self.pos_y[i]) / (math.sqrt((self.pos_x[j] - self.pos_x[i]) ** 2 + (self.pos_y[j] - self.pos_y[i]) ** 2)) ** 3

    def resizeGL(self, width, height):
        gl.glViewport(0, 0, width, height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        aspect = width / float(height)

        GLU.gluPerspective(45.0, aspect, 1.0, 1000.0)
        gl.glMatrixMode(gl.GL_MODELVIEW)

    def paintGL(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        gl.glPushMatrix()

        gl.glTranslate(0.0, 0.0, -990.0)
        gl.glScale(self.scale, self.scale, self.scale)

        for pl in range(self.num_elem):
            gl.glBegin(gl.GL_POLYGON)
            gl.glColor3f(self.color[pl][0], self.color[pl][1], self.color[pl][2])
            for i in range(360):
                theta = i * math.pi / 180
                gl.glVertex2f(self.pos_x[pl] + self.radio[pl] * math.cos(theta),
                              self.pos_y[pl] + self.radio[pl] * math.sin(theta))
            gl.glEnd()

        gl.glPopMatrix()

    def wheelEvent(self, pe):
        if pe.angleDelta().y() > 0:
            self.scale *= 1.1
        elif pe.angleDelta().y() < 0:
            self.scale /= 1.1

        self.glDraw()

    def _ODE(self, r_v, t):
        dr_vdt = []

        c = int(len(r_v) / 4)

        for i in range(2 * c):
            dr_vdt.append(r_v[2 * c + i])

        for i in range(c):
            a_i = 0
            for j in range(c):
                if j != i and i != 0:
                    a_i += self.G * self.mas[j] * (r_v[j] - r_v[i]) / (math.sqrt((r_v[j] - r_v[i]) ** 2 + (r_v[c + j] - r_v[c + i]) ** 2)) ** 3

            dr_vdt.append(a_i)

        for i in range(c):
            a_i = 0
            for j in range(c):
                if j != i and i != 0:
                    a_i += self.G * self.mas[j] * (r_v[c + j] - r_v[c + i]) / (math.sqrt((r_v[j] - r_v[i]) ** 2 + (r_v[c + j] - r_v[c + i]) ** 2)) ** 3

            dr_vdt.append(a_i)

        return dr_vdt

    def Ode_solver(self):
        # self.pos = [[0, 0], [5.7910006, 0], [10.8199995, 0], [14.9599951, 0], [22.793992, 0], [77.8330257, 0], [142.9400028, 0], [287.0989228, 0], [450.4299579, 0]]
        # self.v = [[0, 0], [0, 47.87], [0, 35.02], [0, 29.78], [0, 24.13], [0, 13.07], [0, 9.69], [0, 6.81], [0, 5.43]]
        # for j in range(len(self.v)):
        #     self.v[j] = [i / 10**2 for i in self.v[j]]

        self.initialize()

        r_v_x_y0 = []
        for i in range(self.num_elem):
            r_v_x_y0.append(self.pos_x[i])
        for i in range(self.num_elem):
            r_v_x_y0.append(self.pos_y[i])
        for i in range(self.num_elem):
            r_v_x_y0.append(self.v_x[i])
        for i in range(self.num_elem):
            r_v_x_y0.append(self.v_y[i])

        self.r_v_x_y = odeint(self._ODE, r_v_x_y0, self.time)

        for t_i in range(len(self.time)):
            for i in range(self.num_elem):
                self.pos_x[i] = self.r_v_x_y[t_i][i]
                self.pos_y[i] = self.r_v_x_y[t_i][self.num_elem + i]
            for i in range(self.num_elem):
                self.v_x[i] = self.r_v_x_y[t_i][2 * self.num_elem + i]
                self.v_y[i] = self.r_v_x_y[t_i][2 * self.num_elem + self.num_elem + i]

            # self.glDraw()

        pos_x = []
        pos_y = []
        speed_x = []
        speed_y = []
        for t_i in range(len(self.time)):
            pos_x.append(self.r_v_x_y[t_i][0:self.num_elem + 1])
            pos_y.append(self.r_v_x_y[t_i][self.num_elem + 1:2 * self.num_elem + 1])
            speed_x.append(self.r_v_x_y[t_i][2 * self.num_elem + 1:3 * self.num_elem + 1])
            speed_y.append(self.r_v_x_y[t_i][3 * self.num_elem + 1:4 * self.num_elem + 1])

        self.Sol_ode = np.concatenate((pos_x, pos_y, speed_x, speed_y), axis=-1)

    def python_Verlet(self):
        start_time = time.time()

        self.initialize()

        self.Sol_verl = []

        pos_x_buf = []
        pos_y_buf = []
        speed_x_buf = []
        speed_y_buf = []

        self.pos_x_i = []
        self.pos_y_i = []
        self.speed_x_i = []
        self.speed_y_i = []

        for t_i in range(len(self.time)):
            for i in range(self.num_elem):
                pos_x_buf.append(self.pos_x[i])
                pos_y_buf.append(self.pos_y[i])
                speed_x_buf.append(self.v_x[i])
                speed_y_buf.append(self.v_y[i])
            self.pos_x_i.append(pos_x_buf)
            self.pos_y_i.append(pos_y_buf)
            self.speed_x_i.append(speed_x_buf)
            self.speed_y_i.append(speed_y_buf)

            pos_x_buf = []
            pos_y_buf = []
            speed_x_buf = []
            speed_y_buf = []

            self.a = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
            for i in range(self.num_elem):
                for j in range(self.num_elem):
                    if j != i and i != 0:
                        self.a[i][0] += self.G * self.mas[j] * (self.pos_x[j] - self.pos_x[i]) / \
                                        (math.sqrt((self.pos_x[j] - self.pos_x[i]) ** 2 + (
                                                    self.pos_y[j] - self.pos_y[i]) ** 2)) ** 3

                        self.a[i][1] += self.G * self.mas[j] * (self.pos_y[j] - self.pos_y[i]) / \
                                        (math.sqrt((self.pos_x[j] - self.pos_x[i]) ** 2 + (
                                                    self.pos_y[j] - self.pos_y[i]) ** 2)) ** 3

            for i in range(self.num_elem):
                self.pos_x[i] = self.pos_x[i] + self.v_x[i] * self.delt_t + 0.5 * self.a[i][0] * self.delt_t ** 2
                self.pos_y[i] = self.pos_y[i] + self.v_y[i] * self.delt_t + 0.5 * self.a[i][1] * self.delt_t ** 2

            a_old = self.a
            self.a = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
            for i in range(self.num_elem):
                for j in range(self.num_elem):
                    if j != i and i != 0:
                        self.a[i][0] += self.G * self.mas[j] * (self.pos_x[j] - self.pos_x[i]) / \
                                        (math.sqrt((self.pos_x[j] - self.pos_x[i]) ** 2 + (
                                                    self.pos_y[j] - self.pos_y[i]) ** 2)) ** 3

                        self.a[i][1] += self.G * self.mas[j] * (self.pos_y[j] - self.pos_y[i]) / \
                                        (math.sqrt((self.pos_x[j] - self.pos_x[i]) ** 2 + (
                                                    self.pos_y[j] - self.pos_y[i]) ** 2)) ** 3

            for i in range(self.num_elem):
                self.v_x[i] = self.v_x[i] + 0.5 * (self.a[i][0] + a_old[i][0]) * self.delt_t
                self.v_y[i] = self.v_y[i] + 0.5 * (self.a[i][1] + a_old[i][1]) * self.delt_t

            # self.glDraw()
            # time.sleep(0.1)
        self.Sol_verl = np.concatenate((self.pos_x_i, self.pos_y_i, self.speed_x_i, self.speed_y_i), axis=-1)

        end_time = time.time()
        print(end_time - start_time)

    # ------------------------------------------------------------------

    def V_th(self, v_x, v_y, a, a_old, delt_t, i):
        v_out_x = v_x[i] + 0.5 * (a[i][0] + a_old[i][0]) * delt_t
        v_out_y = v_y[i] + 0.5 * (a[i][1] + a_old[i][1]) * delt_t
        return v_out_x, v_out_y, i

    def Verlet_threading(self):
        start_time = time.time()

        self.initialize()

        self.Sol_verl_thr = []

        pos_x_buf = []
        pos_y_buf = []
        speed_x_buf = []
        speed_y_buf = []

        self.pos_x_i = []
        self.pos_y_i = []
        self.speed_x_i = []
        self.speed_y_i = []

        for t_i in range(len(self.time)):
            for i in range(self.num_elem):
                pos_x_buf.append(self.pos_x[i])
                pos_y_buf.append(self.pos_y[i])
                speed_x_buf.append(self.v_x[i])
                speed_y_buf.append(self.v_y[i])
            self.pos_x_i.append(pos_x_buf)
            self.pos_y_i.append(pos_y_buf)
            self.speed_x_i.append(speed_x_buf)
            self.speed_y_i.append(speed_y_buf)

            pos_x_buf = []
            pos_y_buf = []
            speed_x_buf = []
            speed_y_buf = []

            # threads = []

            with ThreadPoolExecutor(max_workers=self.num_elem) as executor:
                jobs = []
                for i in range(self.num_elem):
                    jobs.append(
                        executor.submit(A_th, pos_x=self.pos_x, pos_y=self.pos_y, G=self.G, mas=self.mas, pl_i=i))

                for job in as_completed(jobs):
                    result_done = job.result()
                    i = result_done[-1]
                    self.a[i][0] = result_done[0]
                    self.a[i][1] = result_done[1]

            for i in range(self.num_elem):
                self.pos_x[i] = self.pos_x[i] + self.v_x[i] * self.delt_t + 0.5 * self.a[i][0] * self.delt_t ** 2
                self.pos_y[i] = self.pos_y[i] + self.v_y[i] * self.delt_t + 0.5 * self.a[i][1] * self.delt_t ** 2

            a_old = self.a
            self.a = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]

            with ThreadPoolExecutor(max_workers=self.num_elem) as executor:
                jobs = []
                for i in range(self.num_elem):
                    jobs.append(
                        executor.submit(A_th, pos_x=self.pos_x, pos_y=self.pos_y, G=self.G, mas=self.mas, pl_i=i))

                for job in as_completed(jobs):
                    result_done = job.result()
                    i = result_done[-1]
                    self.a[i][0] = result_done[0]
                    self.a[i][1] = result_done[1]


            for i in range(self.num_elem):
                self.v_x[i] = self.v_x[i] + 0.5 * (self.a[i][0] + a_old[i][0]) * self.delt_t
                self.v_y[i] = self.v_y[i] + 0.5 * (self.a[i][1] + a_old[i][1]) * self.delt_t

            # self.glDraw()

        self.Sol_verl_thr = np.concatenate((self.pos_x_i, self.pos_y_i, self.speed_x_i, self.speed_y_i), axis=-1)

        end_time = time.time()
        print(end_time - start_time)

    # ------------------------------------------------------------------

    def Verlet_proc(self):
        start_time = time.time()

        self.initialize()

        self.Sol_verl_proc = Verlet_proc(self.num_elem, self.time, self.delt_t, self.pos_x, self.pos_y, self.v_x,
                                         self.v_y, self.a, self.G, self.mas)

        end_time = time.time()
        print(end_time - start_time)

    # ------------------------------------------------------------------

    def Verlet_Cyth(self):
        start_time = time.time()

        self.initialize()

        self.Sol_verl_cyth = []

        pos_x = np.array(self.pos_x).astype(np.double)
        pos_y = np.array(self.pos_y).astype(np.double)
        v_x = np.array(self.v_x).astype(np.double)
        v_y = np.array(self.v_y).astype(np.double)
        mas = np.array(self.mas).astype(np.double)

        a_x = np.zeros(self.num_elem)
        a_y = np.zeros(self.num_elem)


        self.pos_x_i = []
        self.pos_y_i = []
        self.speed_x_i = []
        self.speed_y_i = []

        for t_i in range(len(self.time)):
            # for i in range(self.num_elem):
            #     pos_x_buf.append(self.pos_x[i])
            #     pos_y_buf.append(self.pos_y[i])
            #     speed_x_buf.append(self.v_x[i])
            #     speed_y_buf.append(self.v_y[i])
            # self.pos_x_i.append(pos_x_buf)
            # self.pos_y_i.append(pos_y_buf)
            # self.speed_x_i.append(speed_x_buf)
            # self.speed_y_i.append(speed_y_buf)

            # pos_x_buf = []
            # pos_y_buf = []
            # speed_x_buf = []
            # speed_y_buf = []

            self.pos_x_i.append(np.array(pos_x).astype(np.float32))
            self.pos_y_i.append(np.array(pos_y).astype(np.float32))
            self.speed_x_i.append(np.array(v_x).astype(np.float32))
            self.speed_y_i.append(np.array(v_y).astype(np.float32))

            pos_x, pos_y, a_x, a_y = cythv.Pos(pos_x, pos_y, v_x, v_y, a_x, a_y, mas, self.delt_t, self.num_elem)

            v_x, v_y, a_x, a_y = cythv.Speed(pos_x, pos_y, v_x, v_y, a_x, a_y, mas, self.delt_t, self.num_elem)

            # self.glDraw()

        self.Sol_verl_cyth = np.concatenate((self.pos_x_i, self.pos_y_i, self.speed_x_i, self.speed_y_i), axis=-1)

        end_time = time.time()
        print(end_time - start_time)

    # ------------------------------------------------------------------



    def Verlet_CL(self):

        platform = cl.get_platforms()[0]
        device = platform.get_devices()[0]
        cntxt = cl.Context([device])
        queue = cl.CommandQueue(cntxt)
        queue_speed = cl.CommandQueue(cntxt)

        start_time = time.time()

        self.initialize()

        self.Sol_verl_cl = []

        pos_x = np.array(self.pos_x).astype(np.double)
        pos_y = np.array(self.pos_y).astype(np.double)
        v_x = np.array(self.v_x).astype(np.double)
        v_y = np.array(self.v_y).astype(np.double)
        mas = np.array(self.mas).astype(np.double)

        # pos_x_buf0 = []
        # pos_y_buf0 = []
        # speed_x_buf0 = []
        # speed_y_buf0 = []

        self.pos_x_i = []
        self.pos_y_i = []
        self.speed_x_i = []
        self.speed_y_i = []

        code = """
        void A_cl(__global const double* pos_x, __global const double* pos_y, __global const double* mas, double* res_x, double* res_y, int pl_i, __global int* size)  
        {
            double G = 6.67 * 1e-11;
            res_x[0] = 0.0;
            res_y[0] = 0.0;
            for (int j = 0; j < size[0]; j++)
            {
                if (j != pl_i && pl_i != 0)
                {
                    if (pos_x[j] != pos_x[pl_i])
                    {
                        res_x[0] += G * mas[j] * (pos_x[j] - pos_x[pl_i]) / 
                            ( (sqrt( (pos_x[j] - pos_x[pl_i])*(pos_x[j] - pos_x[pl_i]) + (pos_y[j] - pos_y[pl_i])*(pos_y[j] - pos_y[pl_i]) ))*
                            (sqrt( (pos_x[j] - pos_x[pl_i])*(pos_x[j] - pos_x[pl_i]) + (pos_y[j] - pos_y[pl_i])*(pos_y[j] - pos_y[pl_i]) ))*
                            (sqrt( (pos_x[j] - pos_x[pl_i])*(pos_x[j] - pos_x[pl_i]) + (pos_y[j] - pos_y[pl_i])*(pos_y[j] - pos_y[pl_i]) )) );
                    }

                    if (pos_y[j] != pos_y[pl_i])
                    {
                        res_y[0] += G * mas[j] * (pos_y[j] - pos_y[pl_i]) / 
                            ( (sqrt( (pos_x[j] - pos_x[pl_i])*(pos_x[j] - pos_x[pl_i]) + (pos_y[j] - pos_y[pl_i])*(pos_y[j] - pos_y[pl_i]) ))*
                            (sqrt( (pos_x[j] - pos_x[pl_i])*(pos_x[j] - pos_x[pl_i]) + (pos_y[j] - pos_y[pl_i])*(pos_y[j] - pos_y[pl_i]) ))*
                            (sqrt( (pos_x[j] - pos_x[pl_i])*(pos_x[j] - pos_x[pl_i]) + (pos_y[j] - pos_y[pl_i])*(pos_y[j] - pos_y[pl_i]) )) );
                    }
                }
            }
        }
        __kernel void Pos(__global const double* pos_x, __global const double* pos_y, __global double* a_x, __global double* a_y,
                            __global double* res_pos_x, __global double* res_pos_y, 
                            __global const double* v_x, __global const double* v_y, __global double* mas, __global int* size, __global double* delt_t)
        {
            int i = get_global_id(0);
            double buf_a_x[] = {0};
            double buf_a_y[] = {0};
            A_cl(pos_x, pos_y, mas, buf_a_x, buf_a_y, i, size);
            a_x[i] = buf_a_x[0];
            a_y[i] = buf_a_y[0];
            res_pos_x[i] = pos_x[i] + v_x[i] * delt_t[0] + 0.5 * a_x[i] * delt_t[0] * delt_t[0];
            res_pos_y[i] = pos_y[i] + v_y[i] * delt_t[0] + 0.5 * a_y[i] * delt_t[0] * delt_t[0];
        }
        __kernel void Speed(__global const double* pos_x, __global const double* pos_y, __global double* a_x, __global double* a_y,
                            __global const double* v_x, __global const double* v_y, 
                            __global double* res_v_x, __global double* res_v_y, __global double* mas, __global int* size, __global double* delt_t)
        {
            int i = get_global_id(0);
            double buf_a_x[] = {0};
            double buf_a_y[] = {0};
            A_cl(pos_x, pos_y, mas, buf_a_x, buf_a_y, i, size);
            res_v_x[i] = v_x[i] + 0.5 * (buf_a_x[0] + a_x[i]) * delt_t[0];
            res_v_y[i] = v_y[i] + 0.5 * (buf_a_y[0] + a_y[i]) * delt_t[0];
        }
        """

        bld = cl.Program(cntxt, code).build()

        res_pos_x = cl.Buffer(cntxt, cl.mem_flags.WRITE_ONLY, pos_x.nbytes)
        res_pos_y = cl.Buffer(cntxt, cl.mem_flags.WRITE_ONLY, pos_y.nbytes)
        res_v_x = cl.Buffer(cntxt, cl.mem_flags.WRITE_ONLY, v_x.nbytes)
        res_v_y = cl.Buffer(cntxt, cl.mem_flags.WRITE_ONLY, v_y.nbytes)

        a_x = cl.Buffer(cntxt, cl.mem_flags.READ_WRITE, pos_x.nbytes)
        a_y = cl.Buffer(cntxt, cl.mem_flags.READ_WRITE, pos_y.nbytes)

        mas_buf = cl.Buffer(cntxt, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=mas)
        dt_buf = cl.Buffer(cntxt, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.array([self.delt_t]))
        num_elem_buf = cl.Buffer(cntxt, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                 hostbuf=np.array([self.num_elem]))

        for t_i in range(len(self.time)):
            # for i in range(self.num_elem):
            #     pos_x_buf0.append(self.pos_x[i])
            #     pos_y_buf0.append(self.pos_y[i])
            #     speed_x_buf0.append(self.v_x[i])
            #     speed_y_buf0.append(self.v_y[i])
            # self.pos_x_i.append(pos_x_buf0)
            # self.pos_y_i.append(pos_y_buf0)
            # self.speed_x_i.append(speed_x_buf0)
            # self.speed_y_i.append(speed_y_buf0)

            # pos_x_buf0 = []
            # pos_y_buf0 = []
            # speed_x_buf0 = []
            # speed_y_buf0 = []

            self.pos_x_i.append(np.array(pos_x).astype(np.float32))
            self.pos_y_i.append(np.array(pos_y).astype(np.float32))
            self.speed_x_i.append(np.array(v_x).astype(np.float32))
            self.speed_y_i.append(np.array(v_y).astype(np.float32))

            pos_x_buf = cl.Buffer(cntxt, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=pos_x)
            pos_y_buf = cl.Buffer(cntxt, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=pos_y)
            v_x_buf = cl.Buffer(cntxt, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=v_x)
            v_y_buf = cl.Buffer(cntxt, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=v_y)

            bld.Pos(queue, [self.num_elem], None, pos_x_buf, pos_y_buf, a_x, a_y, res_pos_x, res_pos_y, v_x_buf,
                    v_y_buf, mas_buf, num_elem_buf, dt_buf)

            cl.enqueue_copy(queue, pos_x, res_pos_x)
            cl.enqueue_copy(queue, pos_y, res_pos_y)

            pos_x_buf = cl.Buffer(cntxt, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=pos_x)
            pos_y_buf = cl.Buffer(cntxt, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=pos_y)

            bld.Speed(queue_speed, [self.num_elem], None, pos_x_buf, pos_y_buf, a_x, a_y, v_x_buf, v_y_buf, res_v_x,
                      res_v_y, mas_buf, num_elem_buf, dt_buf)

            cl.enqueue_copy(queue_speed, v_x, res_v_x)
            cl.enqueue_copy(queue_speed, v_y, res_v_y)

            # self.glDraw()

        self.Sol_verl_cl = np.concatenate((self.pos_x_i, self.pos_y_i, self.speed_x_i, self.speed_y_i), axis=-1)

        end_time = time.time()
        print(end_time - start_time)

    # ------------------------------------------------------------------

    def Plot_dif(self):
        plt.plot(np.sqrt(np.sum((self.Sol_ode - self.Sol_verl) ** 2, axis=-1)), 'b', label='Error Verlet')
        plt.plot(np.sqrt(np.sum((self.Sol_ode - self.Sol_verl_thr) ** 2, axis=-1)), 'g',
                 label='Error Verlet with threading')
        plt.plot(np.sqrt(np.sum((self.Sol_ode - self.Sol_verl_proc) ** 2, axis=-1)), 'r',
                 label='Error Verlet with multiprocessing')
        plt.plot(np.sqrt(np.sum((self.Sol_ode - self.Sol_verl_cyth) ** 2, axis=-1)), 'c',
                 label='Error Verlet with Cython')
        plt.plot(np.sqrt(np.sum((self.Sol_ode - self.Sol_verl_cl) ** 2, axis=-1)), 'm',
                 label='Error Verlet with OpenCL')
        plt.legend(loc='best')
        plt.xlabel('t')
        plt.grid()
        plt.show()


    # def Plot_dif(self):
    #     main()



# ------------------------------------------------------------------
def A_th(pos_x, pos_y, G, mas, pl_i):
    a_x = 0
    a_y = 0
    for j in range(len(pos_x)):
        if j != pl_i and pl_i != 0:
            a_x += G * mas[j] * (pos_x[j] - pos_x[pl_i]) / (math.sqrt((pos_x[j] - pos_x[pl_i]) ** 2 + (pos_y[j] - pos_y[pl_i]) ** 2)) ** 3

            a_y += G * mas[j] * (pos_y[j] - pos_y[pl_i]) / (math.sqrt((pos_x[j] - pos_x[pl_i]) ** 2 + (pos_y[j] - pos_y[pl_i]) ** 2)) ** 3
    return a_x, a_y, pl_i


def Pos_prosess(Time, num_elem, init_pos_x, init_pos_y, pos_queue, v_queue, G, mas, delt_t, return_pos_queue):
    pos_x = np.zeros((Time, num_elem))
    pos_y = np.zeros((Time, num_elem))
    pos_x[0], pos_y[0] = init_pos_x, init_pos_y
    for n in range(0, Time - 1):
        a_x, a_y = np.zeros(num_elem), np.zeros(num_elem)
        for i in range(num_elem):
            a_x[i], a_y[i], k = A_th(pos_x[n], pos_y[n], G, mas, i)
        v_x, v_y = v_queue.get()
        for i in range(num_elem):
            pos_x[n + 1, i] = pos_x[n, i] + v_x[i] * delt_t + 0.5 * a_x[i] * delt_t ** 2
            pos_y[n + 1, i] = pos_y[n, i] + v_y[i] * delt_t + 0.5 * a_y[i] * delt_t ** 2
        pos_queue.put([pos_x[n + 1], pos_y[n + 1], a_x, a_y])
    return_pos_queue.put([pos_x, pos_y])


def Speed_prosess(Time, num_elem, init_v_x, init_v_y, pos_queue, speed_queue, G, mas, delt_t, return_speed_queue):
    v_x = np.zeros((Time, num_elem))
    v_y = np.zeros((Time, num_elem))
    v_x[0], v_y[0] = init_v_x, init_v_y
    speed_queue.put([v_x[0], v_y[0]])
    for n in range(0, Time - 1):
        pos_x, pos_y, old_a_x, old_a_y = pos_queue.get()
        for i in range(num_elem):
            a_x, a_y, k = A_th(pos_x, pos_y, G, mas, i)
            v_x[n + 1, i] = v_x[n, i] + 0.5 * (a_x + old_a_x[i]) * delt_t
            v_y[n + 1, i] = v_y[n, i] + 0.5 * (a_y + old_a_y[i]) * delt_t
        speed_queue.put([v_x[n + 1], v_y[n + 1]])
    return_speed_queue.put([v_x, v_y])


def Verlet_proc(num_elem, Time, delt_t, pos_x, pos_y, v_x, v_y, a, G, mas):
    Sol_verl_proc = []

    pos_queue, speed_queue = Queue(), Queue()
    return_pos_queue, return_speed_queue = Queue(), Queue()

    pos_prosess = Process(target=Pos_prosess, args=(
    len(Time), num_elem, pos_x, pos_y, pos_queue, speed_queue, G, mas, delt_t, return_pos_queue))
    speed_prosess = Process(target=Speed_prosess, args=(
    len(Time), num_elem, v_x, v_y, pos_queue, speed_queue, G, mas, delt_t, return_speed_queue))

    pos_prosess.start()
    speed_prosess.start()

    x_pos, y_pos = return_pos_queue.get()
    pos_prosess.join()

    v_x, v_y = return_speed_queue.get()
    speed_prosess.join()

    Sol_verl_proc = np.concatenate((x_pos, y_pos, v_x, v_y), axis=-1)

    return Sol_verl_proc


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)

        self.resize(1500, 900)
        self.setWindowTitle('Task2')

        self.glWidget = GLWidget(self) #GLWidget类

        self.initGUI()

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(20)
        self.timer.timeout.connect(self.glWidget.update)
        self.timer.start()
        self.show()

    def initGUI(self):
        central_widget = QtWidgets.QWidget()
        gui_layout = QtWidgets.QGridLayout()
        central_widget.setLayout(gui_layout)

        self.setCentralWidget(central_widget)

        self.glWidget.setFixedSize(1500, 700)
        gui_layout.addWidget(self.glWidget, 0, 0, 1, 0)

        self.Time_name = QtWidgets.QLabel('Time = ')
        self.Time_name.setFixedSize(450, 10)
        gui_layout.addWidget(self.Time_name, 1, 0, 1, 1)

        self.Time_param = QtWidgets.QLineEdit('100')
        gui_layout.addWidget(self.Time_param, 1, 1, 1, 1)

        self.Time_none = QtWidgets.QPushButton('auto')
        self.Time_none.clicked.connect(self.Widget_auto)
        gui_layout.addWidget(self.Time_none, 1, 2, 1, 1)

        self.Step_name = QtWidgets.QLabel('Step = ') #时间间隔 интервал времени
        self.Step_name.setFixedSize(450, 10)
        gui_layout.addWidget(self.Step_name, 2, 0)

        self.Step_param = QtWidgets.QLineEdit('1')
        gui_layout.addWidget(self.Step_param, 2, 1)

        self.Ode_btn = QtWidgets.QPushButton("Odeint")
        self.Ode_btn.clicked.connect(self.Widget_Ode)
        gui_layout.addWidget(self.Ode_btn, 3, 0,1,1)

        self.Verl_btn = QtWidgets.QPushButton("Verlet method - python")
        self.Verl_btn.clicked.connect(self.Widget_Verlet)
        gui_layout.addWidget(self.Verl_btn, 3, 1,1,1)

        self.Verl_proc_btn = QtWidgets.QPushButton("Verlet method with multiprocess")
        self.Verl_proc_btn.clicked.connect(self.Widget_Verlet_proc)
        gui_layout.addWidget(self.Verl_proc_btn, 3, 2,1,1)

        self.Verl_Cyth_btn = QtWidgets.QPushButton("Verlet method - Cython")
        self.Verl_Cyth_btn.clicked.connect(self.Widget_Verlet_Cyth)
        gui_layout.addWidget(self.Verl_Cyth_btn, 4, 0)

        self.Verl_CL_btn = QtWidgets.QPushButton("Verlet method - OpenCL")
        self.Verl_CL_btn.clicked.connect(self.Widget_Verlet_CL)
        gui_layout.addWidget(self.Verl_CL_btn, 4, 1)

        # self.Plot_btn = QtWidgets.QPushButton("Plot")
        # self.Plot_btn.clicked.connect(self.Plot)
        # gui_layout.addWidget(self.Plot_btn, 4,2)

    def Widget_auto(self):
        self.glWidget.init_auto()

    def Widget_Ode(self):
        self.glWidget.Ode_solver()

    def Widget_Verlet(self):
        self.glWidget.python_Verlet()

    def Widget_Verlet_thr(self):
        self.glWidget.Verlet_threading()

    def Widget_Verlet_proc(self):
        self.glWidget.Verlet_proc()

    def Widget_Verlet_Cyth(self):
        self.glWidget.Verlet_Cyth()

    def Widget_Verlet_CL(self):
        self.glWidget.Verlet_CL()

    # def Plot(self):
    #     self.glWidget.Plot_dif()
    #     self.Plot_dif()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    win = MainWindow()
    win.show()

    sys.exit(app.exec_())
