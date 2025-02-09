import os
import sys
sys.path.insert(0, '../')
import tensorflow as tf
import numpy as np
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.random.seed(1234)
tf.set_random_seed(42)
from PiNNs import PhysicsInformedNN
save_data_path =  "data/"
noise = 0.0
save_model_path = "results/"

try:
    os.mkdir(save_model_path)
except:
    pass

if save_path is not None:
    # Doman bounds
    lb_u = np.array([0.0, 0.0])
    ub_u = np.array([0.11, 800.0])

    lb_v = np.array([-0.11, 0.0])
    ub_v = np.array([0.0, 800.0])

    N0 = 400
    N_b = 2000
    N_f_1 = 4500
    N_f_2 = 4600
    N_f_3 = 3000
    layers = [2]
    n = 10
    for i in range(1, n):
        layers.append(124)

    layers.append(1)

    layers2 = [2]
    n = 6
    for i in range(1, n):
        layers2.append(64)
    layers2.append(1)
    t = np.linspace(0.0, 800.0, 2000).reshape(-1, 1)# data['t'].flatten()[:, None]
    x_u = np.linspace(0.0, 0.11, 500).reshape(-1, 1)# data['x_u'].flatten()[:, None]
    x_v = np.linspace(-0.11, 0.0, 500).reshape(-1, 1)# data['x_v'].flatten()[:, None]
    Exact_gra = np.loadtxt(f"{save_path}/Dgrad.txt").reshape(-1, 1)# data['Dgrad'].flatten()[:, None]
    Exact_u = np.loadtxt(f"{save_path}/U_paper.txt").reshape(-1, 1)# data['U_paper'].flatten()[:, None]
    X_u, T = np.meshgrid(x_u, t)
    X_v, T = np.meshgrid(x_v, t)
    X_u_star = np.hstack((X_u.flatten()[:, None], T.flatten()[:, None]))
    X_v_star = np.hstack((X_v.flatten()[:, None], T.flatten()[:, None]))
    idx_x_u = np.random.choice(x_u.shape[0], N0, replace=False)
    idx_x_v = np.random.choice(x_v.shape[0], N0, replace=False)
    x0_u = x_u[idx_x_u, :]
    x0_v = x_v[idx_x_v, :]
    idx_t = np.random.choice(t.shape[0], N_b, replace=False)
    tb = t[idx_t, :]
    gra = Exact_gra[idx_t, :]
    u0 = Exact_u[idx_t, :]
    lb_u = np.array([0.0, 0.0])
    ub_u = np.array([0.11, 800.0])
    lb_v = np.array([-0.11, 0.0])
    ub_v = np.array([0.0, 800.0])
    rtemp1 = lhs(2, N_f_1)
    rtemp2 = lhs(2, N_f_1)
    X_f_u_1 = lb_u + (ub_u - lb_u) * rtemp1
    X_f_v_1 = lb_v + (ub_v - lb_v) * rtemp2
    lb_u = np.array([0.0, 0.0])
    ub_u = np.array([0.05, 800.0])
    lb_v = np.array([-0.05, 0.0])
    ub_v = np.array([0.0, 800.0])
    X_f_u_2 = lb_u + (ub_u - lb_u) * lhs(2, N_f_2)
    X_f_v_2 = lb_v + (ub_v - lb_v) * lhs(2, N_f_2)
    X_f_u = np.concatenate((X_f_u_1, X_f_u_2), 0)
    X_f_v = np.concatenate((X_f_v_1, X_f_v_2), 0)
    lb_u = np.array([0.0, 0.0])
    ub_u = np.array([0.11, 800.0])
    lb_v = np.array([-0.11, 0.0])
    ub_v = np.array([0.0, 800.0])

    model = PhysicsInformedNN(x0_u, x0_v, tb, X_f_u, X_f_v, layers, layers2, lb_u, ub_u, lb_v, ub_v, gra, u0, N_b)

    model.train(2200)#15200
#    model.train(11400)

    X_mid_u = np.concatenate((0 * t, t), 1)
    X_mid_v = np.concatenate((0 * t, t), 1)
    u_mid, v_mid, u_x_mid, v_x_mid, _, _ = model.predict(X_mid_u, X_mid_v)

    t = np.linspace(0.0, 800.0, 200).reshape(-1, 1)# data['t_res'].flatten()[:, None]
    x_u = np.linspace(0.0, 0.11, 200).reshape(-1, 1)# data['x_u'].flatten()[:, None]
    x_v = np.linspace(-0.11, 0.0, 200).reshape(-1, 1)# data['x_v'].flatten()[:, None]
    X_u, T = np.meshgrid(x_u, t)
    X_v, T = np.meshgrid(x_v, t)

    X_u_star = np.hstack((X_u.flatten()[:, None], T.flatten()[:, None]))
    X_v_star = np.hstack((X_v.flatten()[:, None], T.flatten()[:, None]))
    u_pred, v_pred, f_u_pred, f_v_pred, _, _ = model.predict(X_u_star, X_v_star)
    U_pred = griddata(X_u_star, u_pred.flatten(), (X_u, T), method='cubic')
    V_pred = griddata(X_v_star, v_pred.flatten(), (X_v, T), method='cubic')

    np.savetxt(f"{save_path}/upred2.txt", U_pred)
    np.savetxt(f"{save_path}/vpred2.txt", V_pred)
    np.savetxt(f"{save_path}/umid2.txt", u_mid)
    np.savetxt(f"{save_path}/vmid2.txt", v_mid)
    np.savetxt(f"{save_path}/uxmid2.txt", u_x_mid)
    np.savetxt(f"{save_path}/vxmid2.txt", v_x_mid)
    np.savetxt(f"{save_path}/tb.txt", tb)