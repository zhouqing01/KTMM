import numpy as np
import math

import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import Process, Queue
from scipy.integrate import odeint
import sys
import time

import cythv

import pyopencl as cl


G = 6.67 / 10 ** 11
delt_t = 0.1


def initialize(count):
    num_elem = count
    t = 10
    Time = int(t / delt_t)

    pos_x = np.random.random(count)
    # 生成count数个随机数
    pos_y = np.random.random(count)
    v_x = np.random.random(count)
    v_y = np.random.random(count)

    a_x = np.random.random(count)
    a_y = np.random.random(count)
    a = []
    for i in range(count):
        a.append([a_x[i], a_y[i]])

    mas = np.random.random(count)

    return num_elem, Time, pos_x, pos_y, v_x, v_y, a, mas


def A_th(pos_x, pos_y, mas, pl_i, size):
    a_x = 0
    a_y = 0
    for j in range(size):
        if j != pl_i:
            if pos_x[j] != pos_x[pl_i]:
                a_x += G * mas[j] * (pos_x[j] - pos_x[pl_i]) / (np.sqrt((pos_x[j] - pos_x[pl_i]) ** 2 + (pos_y[j] - pos_y[pl_i]) ** 2)) ** 3

            if pos_y[j] != pos_y[pl_i]:
                a_y += G * mas[j] * (pos_y[j] - pos_y[pl_i]) / (np.sqrt((pos_x[j] - pos_x[pl_i]) ** 2 + (pos_y[j] - pos_y[pl_i]) ** 2)) ** 3
    return a_x, a_y, pl_i


def Verlet(num_elem, Time, pos_x, pos_y, v_x, v_y, a, mas):
    start_time = time.time()

    for t_i in range(Time):

        for i in range(num_elem):
            a[i][0], a[i][1], k = A_th(pos_x, pos_y, mas, i, num_elem)

        for i in range(num_elem):
            pos_x[i] = pos_x[i] + v_x[i] * delt_t + 0.5 * a[i][0] * delt_t ** 2
            pos_y[i] = pos_y[i] + v_y[i] * delt_t + 0.5 * a[i][1] * delt_t ** 2

        a_old = a
        for i in range(num_elem):
            a[i][0], a[i][1], k = A_th(pos_x, pos_y, mas, i, num_elem)

        for i in range(num_elem):
            v_x[i] = v_x[i] + 0.5 * (a[i][0] + a_old[i][0]) * delt_t
            v_y[i] = v_y[i] + 0.5 * (a[i][1] + a_old[i][1]) * delt_t

    # Sol_verl = np.concatenate((pos_x_i, pos_y_i, speed_x_i, speed_y_i), axis=-1)

    end_time = time.time()
    return end_time - start_time


# ------------------------------------------------------------------

def Verlet_thr(num_elem, Time, pos_x, pos_y, v_x, v_y, a, mas):
    start_time = time.time()

    a_x = np.zeros(num_elem)
    a_y = np.zeros(num_elem)

    # pos_x_i = []
    # pos_y_i = []
    # speed_x_i = []
    # speed_y_i = []

    for t_i in range(Time):
        # pos_x_i.append(pos_x)
        # pos_y_i.append(pos_y)
        # speed_x_i.append(v_x)
        # speed_y_i.append(v_x)

        with ThreadPoolExecutor(max_workers=num_elem) as executor:
            jobs = []
            for i in range(num_elem):
                jobs.append(executor.submit(A_th, pos_x=pos_x, pos_y=pos_y, mas=mas, pl_i=i, size=num_elem))

            for job in as_completed(jobs):
                result_done = job.result()
                i = result_done[-1]
                a_x[i] = result_done[0]
                a_y[i] = result_done[1]

        with ThreadPoolExecutor(max_workers=num_elem) as executor:
            jobs = []
            for i in range(num_elem):
                jobs.append(executor.submit(Pos, pos_x=pos_x, pos_y=pos_y, v_x=v_x, v_y=v_y, a_x=a_x, a_y=a_y, mas=mas,
                                            delt_t=delt_t, size=num_elem, i=i))

            for job in as_completed(jobs):
                result_done = job.result()
                i = result_done[-1]
                pos_x[i] = result_done[0]
                pos_y[i] = result_done[1]

        with ThreadPoolExecutor(max_workers=num_elem) as executor:
            jobs = []
            for i in range(num_elem):
                jobs.append(
                    executor.submit(Speed, pos_x=pos_x, pos_y=pos_y, v_x=v_x, v_y=v_y, a_x=a_x, a_y=a_y, mas=mas,
                                    delt_t=delt_t, size=num_elem, i=i))

            for job in as_completed(jobs):
                result_done = job.result()
                i = result_done[-1]
                v_x[i] = result_done[0]
                v_y[i] = result_done[1]
                a_x[i] = result_done[2]
                a_y[i] = result_done[3]

    # Sol_verl_thr = np.concatenate((pos_x_i, pos_y_i, speed_x_i, speed_y_i), axis=-1)

    end_time = time.time()
    return end_time - start_time

# ------------------------------------------------------------------

def Verlet_ode(num_elem, Time, pos_x, pos_y, v_x, v_y, a, mas):
    start_time = time.time()

    def problemPosition(v, t):
        res = np.zeros(v.size)
        for i in range(v.size):
            res[i] = v[i]
        return res

    def problemVelocity(x, t):
        res = np.zeros(x.size)

        positions = np.zeros((Time.size, x.size))
        positions[0] = x

        for i in range(x.size // 2):
            res[2 * i:2 * i + 2] = A_th(i, mas, positions)[i]
        return res

    positions = odeint(problemPosition, [pos_x, pos_y], Time)
    velocities = odeint(problemVelocity, [v_x, v_y], Time)

    end_time = time.time()
    return end_time - start_time

# ------------------------------------------------------------------

def Pos_prosess(Time, num_elem, init_pos_x, init_pos_y, pos_queue, speed_queue, mas, return_pos_queue):
    pos_x = np.zeros((Time, num_elem))
    pos_y = np.zeros((Time, num_elem))

    pos_x[0], pos_y[0] = init_pos_x, init_pos_y

    for n in range(0, Time - 1):
        # v_x, v_y, a_x, a_y = speed_queue.get()

        a_x, a_y = np.zeros(num_elem), np.zeros(num_elem)
        for i in range(num_elem):
            a_x[i], a_y[i], k = A_th(pos_x[n], pos_y[n], mas, i, num_elem)
        v_x, v_y = speed_queue.get()

        for i in range(num_elem):
            pos_x[n + 1, i] = pos_x[n, i] + v_x[i] * delt_t + 0.5 * a_x[i] * delt_t ** 2
            pos_y[n + 1, i] = pos_y[n, i] + v_y[i] * delt_t + 0.5 * a_y[i] * delt_t ** 2
        pos_queue.put([pos_x[n + 1], pos_y[n + 1], a_x, a_y])
    return_pos_queue.put([pos_x, pos_y])


def Speed_prosess(Time, num_elem, init_v_x, init_v_y, init_a, pos_queue, speed_queue, mas, return_speed_queue):
    v_x = np.zeros((Time, num_elem))
    v_y = np.zeros((Time, num_elem))

    v_x[0], v_y[0] = init_v_x, init_v_y
    # a_x, a_y = init_a[:, 0], init_a[:, 1]

    # speed_queue.put([v_x[0], v_y[0], a_x, a_y])
    speed_queue.put([v_x[0], v_y[0]])
    for n in range(0, Time - 1):
        pos_x, pos_y, old_a_x, old_a_y = pos_queue.get()

        # a_x, a_y = np.zeros(num_elem), np.zeros(num_elem)
        # for i in range(num_elem):
        #     a_x[i], a_y[i], k = A_th(pos_x, pos_y, mas, i, num_elem)

        for i in range(num_elem):
            a_x, a_y, k = A_th(pos_x, pos_y, mas, i, num_elem)
            v_x[n + 1, i] = v_x[n, i] + 0.5 * (a_x + old_a_x[i]) * delt_t
            v_y[n + 1, i] = v_y[n, i] + 0.5 * (a_y + old_a_y[i]) * delt_t
            # v_x[n + 1, i] = v_x[n, i]  + 0.5 * (a_x[i] + old_a_x[i]) * delt_t
            # v_y[n + 1, i] = v_y[n, i]  + 0.5 * (a_y[i] + old_a_y[i]) * delt_t
        # speed_queue.put([v_x[n + 1], v_y[n + 1], a_x, a_y])
        speed_queue.put([v_x[n + 1], v_y[n + 1]])
    return_speed_queue.put([v_x, v_y])


def Verlet_proc(num_elem, Time, pos_x, pos_y, v_x, v_y, a, mas):
    start_time = time.time()

    a = np.array(a)

    pos_queue, speed_queue = Queue(), Queue()
    return_pos_queue, return_speed_queue = Queue(), Queue()

    pos_prosess = Process(target=Pos_prosess,
                          args=(Time, num_elem, pos_x, pos_y, pos_queue, speed_queue, mas, return_pos_queue))
    speed_prosess = Process(target=Speed_prosess,
                            args=(Time, num_elem, v_x, v_y, a, pos_queue, speed_queue, mas, return_speed_queue))

    pos_prosess.start()
    speed_prosess.start()

    x_pos, y_pos = return_pos_queue.get()
    pos_prosess.join()

    v_x, v_y = return_speed_queue.get()
    speed_prosess.join()

    # Sol_verl_proc = np.concatenate((x_pos, y_pos, v_x, v_y), axis=-1)

    end_time = time.time()
    return end_time - start_time


def Pos(pos_x, pos_y, v_x, v_y, a_x, a_y, mas, delt_t, size, i):
    out_x = pos_x[i] + v_x[i] * delt_t + 0.5 * a_x[i] * delt_t ** 2
    out_y = pos_y[i] + v_y[i] * delt_t + 0.5 * a_y[i] * delt_t ** 2

    return out_x, out_y, i


def Speed(pos_x, pos_y, v_x, v_y, a_x, a_y, mas, delt_t, size, i):
    a = A_th(pos_x, pos_y, mas, i, size)
    a_x_next = a[0]
    a_y_next = a[1]
    out_x = v_x[i] + 0.5 * (a_x_next + a_x[i]) * delt_t
    out_y = v_y[i] + 0.5 * (a_y_next + a_y[i]) * delt_t

    return out_x, out_y, a_x_next, a_y_next, i


def Verlet_proc_pool(num_elem, Time, pos_x, pos_y, v_x, v_y, a, mas):
    start_time = time.time()

    a_x = np.zeros(num_elem)
    a_y = np.zeros(num_elem)

    # pos_x_i = []
    # pos_y_i = []
    # speed_x_i = []
    # speed_y_i = []

    with ProcessPoolExecutor(max_workers=4) as executor:
        jobs = []
        for i in range(num_elem):
            jobs.append(executor.submit(A_th, pos_x=pos_x, pos_y=pos_y, mas=mas, pl_i=i, size=num_elem))

        for job in as_completed(jobs):
            result_done = job.result()
            i = result_done[-1]
            a_x[i] = result_done[0]
            a_y[i] = result_done[1]

        for t_i in range(Time):
            # pos_x_i.append(pos_x)
            # pos_y_i.append(pos_y)
            # speed_x_i.append(v_x)
            # speed_y_i.append(v_x)

            jobs = []
            for i in range(num_elem):
                jobs.append(executor.submit(Pos, pos_x=pos_x, pos_y=pos_y, v_x=v_x, v_y=v_y, a_x=a_x, a_y=a_y, mas=mas,
                                            delt_t=delt_t, size=num_elem, i=i))

            for job in as_completed(jobs):
                result_done = job.result()
                i = result_done[-1]
                pos_x[i] = result_done[0]
                pos_y[i] = result_done[1]

            jobs = []
            for i in range(num_elem):
                jobs.append(
                    executor.submit(Speed, pos_x=pos_x, pos_y=pos_y, v_x=v_x, v_y=v_y, a_x=a_x, a_y=a_y, mas=mas,
                                    delt_t=delt_t, size=num_elem, i=i))

            for job in as_completed(jobs):
                result_done = job.result()
                i = result_done[-1]
                v_x[i] = result_done[0]
                v_y[i] = result_done[1]
                a_x[i] = result_done[2]
                a_y[i] = result_done[3]

    # Sol_verl_proc = np.concatenate((pos_x_i, pos_y_i, speed_x_i, speed_y_i), axis=-1)

    end_time = time.time()
    return end_time - start_time


# ------------------------------------------------------------------

def Verlet_Cyth(num_elem, Time, pos_x, pos_y, v_x, v_y, a, mas):
    start_time = time.time()

    a_x = np.zeros(num_elem)
    a_y = np.zeros(num_elem)

    # pos_x_i = []
    # pos_y_i = []
    # speed_x_i = []
    # speed_y_i = []

    for t_i in range(Time):
        # pos_x_i.append(pos_x)
        # pos_y_i.append(pos_y)
        # speed_x_i.append(v_x)
        # speed_y_i.append(v_x)

        pos_x, pos_y, a_x, a_y = cythv.Pos(pos_x, pos_y, v_x, v_y, a_x, a_y, mas, delt_t, num_elem)

        v_x, v_y, a_x, a_y = cythv.Speed(pos_x, pos_y, v_x, v_y, a_x, a_y, mas, delt_t, num_elem)

    # Sol_verl_cyth = np.concatenate((pos_x_i, pos_y_i, speed_x_i, speed_y_i), axis=-1)

    end_time = time.time()
    return end_time - start_time


# ------------------------------------------------------------------

def Verlet_CL(num_elem, Time, pos_x, pos_y, v_x, v_y, a, mas, N1, N2):
    platform = cl.get_platforms()[N1]
    device = platform.get_devices()[N2]
    cntxt = cl.Context([device])
    queue = cl.CommandQueue(cntxt)
    queue_speed = cl.CommandQueue(cntxt)

    pos_x = np.array(pos_x).astype(np.double)
    pos_y = np.array(pos_y).astype(np.double)
    v_x = np.array(v_x).astype(np.double)
    v_y = np.array(v_y).astype(np.double)

    start_time = time.time()

    # pos_x_i = []
    # pos_y_i = []
    # speed_x_i = []
    # speed_y_i = []

    code = """
    void A_cl(__global const double* pos_x, __global const double* pos_y, __global const double* mas, double* res_x, double* res_y, int pl_i, __global int* size)  
    {
        double G = 6.67 * 1e-11;
        res_x[0] = 0.0;
        res_y[0] = 0.0;
        for (int j = 0; j < size[0]; j++)
        {
            if (j != pl_i)
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
    dt_buf = cl.Buffer(cntxt, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.array([delt_t]))
    num_elem_buf = cl.Buffer(cntxt, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.array([num_elem]))

    for t_i in range(Time):

        pos_x_buf = cl.Buffer(cntxt, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=pos_x)
        pos_y_buf = cl.Buffer(cntxt, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=pos_y)
        v_x_buf = cl.Buffer(cntxt, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=v_x)
        v_y_buf = cl.Buffer(cntxt, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=v_y)

        bld.Pos(queue, [num_elem], None, pos_x_buf, pos_y_buf, a_x, a_y, res_pos_x, res_pos_y, v_x_buf, v_y_buf, mas_buf, num_elem_buf, dt_buf)

        cl.enqueue_copy(queue, pos_x, res_pos_x)
        cl.enqueue_copy(queue, pos_y, res_pos_y)

        pos_x_buf = cl.Buffer(cntxt, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=pos_x)
        pos_y_buf = cl.Buffer(cntxt, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=pos_y)

        bld.Speed(queue_speed, [num_elem], None, pos_x_buf, pos_y_buf, a_x, a_y, v_x_buf, v_y_buf, res_v_x, res_v_y, mas_buf, num_elem_buf, dt_buf)

        cl.enqueue_copy(queue_speed, v_x, res_v_x)
        cl.enqueue_copy(queue_speed, v_y, res_v_y)

    # Sol_verl_cl = np.concatenate((pos_x_i, pos_y_i, speed_x_i, speed_y_i), axis=-1)

    end_time = time.time()
    return end_time - start_time


# ------------------------------------------------------------------

def Plot(count, time_Verl, time_Verl_ode, time_Verl_proc, time_Verl_proc_pool, time_Verl_Cyth, time_Verl_CL):
    #plt.subplot(1, 2, 1)
    plt.plot(count, time_Verl, 'b', label='Verlet')
    plt.plot(count, time_Verl_ode, 'g', label='Verlet with odeint')
    # plt.plot(count_max, time_Verl_proc, 'y', label='Verlet with multiprocessing')
    plt.plot(count, time_Verl_proc_pool, 'r', label='Verlet with multiprocessing')
    plt.plot(count, time_Verl_Cyth, 'c', label='Verlet with Cython')
    plt.plot(count, time_Verl_CL, 'm', label='Verlet with OpenCL')

    # plt.plot(count_max, time_Verl_CL00, 'r', label='Verlet with OpenCL NVIDIA CUDA')
    # # plt.plot(count_max, time_Verl_CL10, 'g', label='Verlet with OpenCL Intel(R) HD Graphics 4600')
    # plt.plot(count_max, time_Verl_CL11, 'b', label='Verlet with OpenCL Intel(R) Core(TM) i7-4710HQ CPU @ 2.50GHz')
    plt.legend(loc='best')
    plt.xlabel('N')
    plt.ylabel('time')
    plt.grid()
    plt.show()

    #plt.subplot(1, 2, 2)

    plt.plot(count, time_Verl / time_Verl_ode, 'g', label='Verlet with odeint')
    # plt.plot(count_max, time_Verl / time_Verl_proc, 'y', label='Verlet with multiprocessing')
    plt.plot(count, time_Verl / time_Verl_proc_pool, 'r', label='Verlet with multiprocessing')
    plt.plot(count, time_Verl / time_Verl_Cyth, 'c', label='Verlet with Cython')
    plt.plot(count, time_Verl / time_Verl_CL, 'm', label='Verlet with OpenCL')

    # # plt.plot(count_max, time_Verl / time_Verl_CL00, 'r', label='Verlet with OpenCL NVIDIA CUDA')
    # # plt.plot(count_max, time_Verl / time_Verl_CL10, 'g', label='Verlet with OpenCL Intel(R) HD Graphics 4600')
    # # plt.plot(count_max, time_Verl / time_Verl_CL11, 'b', label='Verlet with OpenCL Intel(R) Core(TM) i7-4710HQ CPU @ 2.50GHz')
    plt.legend(loc='best')
    plt.xlabel('N')
    plt.ylabel('time_verl/time')
    plt.grid()

    plt.show()


# ------------------------------------------------------------------


def main():
    # count_max = [50, 100, 200]
    count = [50, 100, 200]
    repeat = 1

    time_Verl = np.zeros(len(count))
    time_Verl_ode = np.zeros(len(count))
    time_Verl_proc = np.zeros(len(count))
    time_Verl_proc_pool = np.zeros(len(count))
    time_Verl_Cyth = np.zeros(len(count))
    time_Verl_CL = np.zeros(len(count))

    # time_Verl_CL00 = np.zeros(len(count_max))  # NVIDIA CUDA
    # time_Verl_CL10 = np.zeros(len(count_max))  # Intel(R) HD Graphics 4600
    # time_Verl_CL11 = np.zeros(len(count_max))  # Intel(R) Core(TM) i7-4710HQ CPU @ 2.50GHz

    for c_i in range(len(count)):
        num_elem, Time, pos_x, pos_y, v_x, v_y, a, mas = initialize(count[c_i])

        # for i in range(repeat):
        #     time_Verl_thr[c_i] += Verlet_thr(num_elem, Time, pos_x, pos_y, v_x, v_y, a, mas)
        # time_Verl_thr[c_i] /= repeat
        # print("Verlet with threading ", " c_i= ", c_i, " time: ", time_Verl_thr[c_i])

        for i in range(repeat):
            time_Verl_ode[c_i] += Verlet_thr(num_elem, Time, pos_x, pos_y, v_x, v_y, a, mas)
        time_Verl_ode[c_i] /= repeat
        print("Verlet with odeint ", " c_i= ", c_i, " time: ", time_Verl_ode[c_i])

        # for i in range(repeat):
        #     time_Verl_proc[c_i] += Verlet_proc(num_elem, Time, pos_x, pos_y, v_x, v_y, a, mas)
        # time_Verl_proc[c_i] /= repeat
        # print("Verlet with multiprocessing ", " c_i= ", c_i, " time: ", time_Verl_proc[c_i])

        for i in range(repeat):
            time_Verl_proc_pool[c_i] += Verlet_proc_pool(num_elem, Time, pos_x, pos_y, v_x, v_y, a, mas)
        time_Verl_proc_pool[c_i] /= repeat
        print("Verlet with multiprocessing", " c_i= ", c_i, " time: ", time_Verl_proc_pool[c_i])

        for i in range(repeat):
            time_Verl_Cyth[c_i] += Verlet_Cyth(num_elem, Time, pos_x, pos_y, v_x, v_y, a, mas)
        time_Verl_Cyth[c_i] /= repeat
        print("Verlet with Cython ", " c_i= ", c_i, " time: ", time_Verl_Cyth[c_i])

        for i in range(repeat):
            time_Verl_CL[c_i] += Verlet_CL(num_elem, Time, pos_x, pos_y, v_x, v_y, a, mas, 0, 0)
        time_Verl_CL[c_i] /= repeat
        print("Verlet with OpenCL ", " c_i= ", c_i, " time: ", time_Verl_CL[c_i])

        for i in range(repeat):
            time_Verl[c_i] += Verlet(num_elem, Time, pos_x, pos_y, v_x, v_y, a, mas)
        time_Verl[c_i] /= repeat
        print("Verlet with python ", " c_i= ", c_i, " time: ", time_Verl[c_i])

        # for i in range(repeat):
        #     time_Verl_CL00[c_i] += Verlet_CL(num_elem, Time, pos_x, pos_y, v_x, v_y, a, mas, 0, 0)
        # time_Verl_CL00[c_i] /= repeat
        # print("Verlet with OpenCL NVIDIA CUDA ", " c_i= ", c_i, " time: ", time_Verl_CL00[c_i])

        # # for i in range(repeat):
        # #     time_Verl_CL10[c_i] += Verlet_CL(num_elem, Time, pos_x, pos_y, v_x, v_y, a, mas, 1, 0)
        # # time_Verl_CL10[c_i] /= repeat
        # # print("Verlet with OpenCL Intel(R) HD Graphics 4600 ", " c_i= ", c_i, " time: ", time_Verl_CL10[c_i])

        # for i in range(repeat):
        #     time_Verl_CL11[c_i] += Verlet_CL(num_elem, Time, pos_x, pos_y, v_x, v_y, a, mas, 1, 1)
        # time_Verl_CL11[c_i] /= repeat
        # print("Verlet with OpenCL Intel(R) Core(TM) i7-4710HQ CPU @ 2.50GHz ", " c_i= ", c_i, " time: ", time_Verl_CL11[c_i])

        print()

    # Plot(count_max, time_Verl, time_Verl_thr, time_Verl_proc, time_Verl_proc_pool, time_Verl_Cyth, time_Verl_CL,
    #      time_Verl_CL00, time_Verl_CL10, time_Verl_CL11)
    Plot(count, time_Verl, time_Verl_ode, time_Verl_proc, time_Verl_proc_pool, time_Verl_Cyth, time_Verl_CL)


if __name__ == '__main__':
    main()