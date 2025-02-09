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

class PhysicsInformedNN:
    """
    A physics-informed neural network (PINN) class for solving partial differential equations.
    This implementation maintains the original variable names and logic structure.
    """

    # === Initialization ===
    def __init__(self, x0_u, x0_v, tb, X_f_u, X_f_v, layers, layers2, lb_u, ub_u, lb_v, ub_v, gra, u0, N_b):
        """
        Initialize the PINN model with input data and network architecture parameters.
        """
        # Preprocess input data
        X0_u = np.concatenate((x0_u, 0 * x0_u), 1)  # (x0, 0)
        X0_v = np.concatenate((x0_v, 0 * x0_v), 1)  # (x0, 0)
        X_mid = np.concatenate((0 * tb, tb), 1)  # (0, tb)
        X_lb_v = np.concatenate((0 * tb + lb_v[0], tb), 1)  # (lb[0], tb)
        X_ub_u = np.concatenate((0 * tb + ub_u[0], tb), 1)  # (ub[0], tb)

        # Store boundary and domain information
        self.lb_v = lb_v
        self.ub_v = ub_v
        self.lb_u = lb_u
        self.ub_u = ub_u

        # Extract spatial and temporal components
        self.x0_u = X0_u[:, 0:1]
        self.t0_u = X0_u[:, 1:2]
        self.x0_v = X0_v[:, 0:1]
        self.t0_v = X0_v[:, 1:2]
        self.x_lb_v = X_lb_v[:, 0:1]
        self.t_lb_v = X_lb_v[:, 1:2]
        self.x_ub_u = X_ub_u[:, 0:1]
        self.t_ub_u = X_ub_u[:, 1:2]
        self.x_mid = X_mid[:, 0:1]
        self.t_mid = X_mid[:, 1:2]

        # Collocation points
        self.x_f_u = X_f_u[:, 0:1]
        self.t_f_u = X_f_u[:, 1:2]
        self.x_f_v = X_f_v[:, 0:1]
        self.t_f_v = X_f_v[:, 1:2]

        # Midpoint conditions
        self.x_mid_end = np.expand_dims(np.array([lb_u[0]]), axis=1)
        self.t_mid_end = np.expand_dims(np.array([ub_u[1]]), axis=1)

        # Additional parameters
        self.gra = gra
        self.u0 = u0
        self.layers = layers
        self.layers2 = layers2

        # TensorFlow constants and variables
        with tf.variable_scope("temp"):
            self.D1_log = tf.constant(-13.8155, name='D1')
            self.D2_log = tf.constant(-23.0259, name='D2')
            self.k_log = tf.Variable(-13.8155, name='k')
            self.E = tf.Variable(0.3, name='E')
            self.C = tf.constant(2.0)
            self.A = tf.constant(0.16, name='A')
            self.alpha = tf.Variable(0.9, name='alpha')

            self.u_mid_fix = tf.Variable(0.0 * tf.ones([N_b, 1]), dtype=tf.float32)
            self.v_mid_fix = tf.Variable(0.0 * tf.ones([N_b, 1]), dtype=tf.float32)

            self.power_u_log = tf.constant(0.0)
            self.power_v_log = tf.Variable(3.0)

            self.k = tf.exp(self.k_log)
            self.power_u = tf.exp(self.power_u_log)
            self.power_v = tf.exp(self.power_v_log)
            self.D1 = tf.exp(1.0 * self.D1_log)
            self.D2 = tf.exp(1.0 * self.D2_log)

        # Neural network weights and biases
        self.weights_u, self.biases_u = self.initialize_NN(layers, 0)
        self.weights_v, self.biases_v = self.initialize_NN(layers, 1)
        self.weights_D, self.biases_D = self.initialize_NN(layers2, 0)
        self.weights_u_D, self.biases_u_D = self.initialize_NN(layers2, 0)

        # Placeholders for input data
        self.x0_u_tf = tf.placeholder(tf.float32, shape=[None, self.x0_u.shape[1]], name='input_x')
        self.t0_u_tf = tf.placeholder(tf.float32, shape=[None, self.t0_u.shape[1]], name='input_t')
        self.x0_v_tf = tf.placeholder(tf.float32, shape=[None, self.x0_v.shape[1]])
        self.t0_v_tf = tf.placeholder(tf.float32, shape=[None, self.t0_v.shape[1]])
        self.gra_tf = tf.placeholder(tf.float32, shape=[None, self.gra.shape[1]])
        self.x_lb_v_tf = tf.placeholder(tf.float32, shape=[None, self.x_lb_v.shape[1]])
        self.t_lb_v_tf = tf.placeholder(tf.float32, shape=[None, self.t_lb_v.shape[1]])
        self.x_ub_u_tf = tf.placeholder(tf.float32, shape=[None, self.x_ub_u.shape[1]])
        self.t_ub_u_tf = tf.placeholder(tf.float32, shape=[None, self.t_ub_u.shape[1]])
        self.x_mid_tf = tf.placeholder(tf.float32, shape=[None, self.x_mid.shape[1]])
        self.t_mid_tf = tf.placeholder(tf.float32, shape=[None, self.t_mid.shape[1]])
        self.x_f_u_tf = tf.placeholder(tf.float32, shape=[None, self.x_f_u.shape[1]])
        self.t_f_u_tf = tf.placeholder(tf.float32, shape=[None, self.t_f_u.shape[1]])
        self.x_f_v_tf = tf.placeholder(tf.float32, shape=[None, self.x_f_v.shape[1]])
        self.t_f_v_tf = tf.placeholder(tf.float32, shape=[None, self.t_f_v.shape[1]])
        self.x_mid_end_tf = tf.placeholder(tf.float32, shape=[None, self.x_mid_end.shape[1]])
        self.t_mid_end_tf = tf.placeholder(tf.float32, shape=[None, self.t_mid_end.shape[1]])

        # Define computational graphs
        self.u0_pred, _, _, _ = self.net_uv(self.x0_u_tf, self.t0_u_tf)
        _, _, self.u0_x_pred, _ = self.net_uv(self.x0_u_tf, self.t0_u_tf)
        _, self.v0_pred, _, _ = self.net_uv(self.x0_v_tf, self.t0_v_tf)
        _, _, _, self.v0_x_pred = self.net_uv(self.x0_v_tf, self.t0_v_tf)
        _, self.v_lb_pred, _, _ = self.net_uv(self.x_lb_v_tf, self.t_lb_v_tf)
        _, _, _, self.v_x_lb_pred = self.net_uv(self.x_lb_v_tf, self.t_lb_v_tf)
        self.u_ub_pred, _, _, _ = self.net_uv(self.x_ub_u_tf, self.t_ub_u_tf)
        _, _, self.u_x_ub_pred, _ = self.net_uv(self.x_ub_u_tf, self.t_ub_u_tf)
        self.u_mid_pred, _, _, _ = self.net_uv(self.x_mid_tf, self.t_mid_tf)
        _, self.v_mid_pred, _, _ = self.net_uv(self.x_mid_tf, self.t_mid_tf)
        _, _, self.u_x_mid_pred, _ = self.net_uv(self.x_mid_tf, self.t_mid_tf)
        _, _, _, self.v_x_mid_pred = self.net_uv(self.x_mid_tf, self.t_mid_tf)
        self.f_u_pred, _, _, _ = self.net_f_uv(self.x_f_u_tf, self.t_f_u_tf)
        _, self.f_v_pred, _, _ = self.net_f_uv(self.x_f_v_tf, self.t_f_v_tf)

        # Post-processing predictions
        self.u_mid_pred_fin = self.u_mid_pred + 1e-2 * tf.tanh(self.u_mid_fix)
        self.v_mid_pred_fin = self.v_mid_pred + 1e-2 * tf.tanh(self.v_mid_fix)

        # Loss terms
        self.Ex1 = tf.exp(
            1.0 * self.alpha * 96485.0 / 8.314 / 298.15 * (tf.abs(2.0 - 5e-3 * self.t_mid_tf) - self.E))
        self.Ex2 = tf.exp(
            1.0 * (1.0 - self.alpha) * 96485.0 / 8.314 / 298.15 * (tf.abs(2.0 - 5e-3 * self.t_mid_tf) - self.E))
        self.Ex3 = tf.exp(96485.0 / 8.314 / 298.15 * (tf.abs(2.0 - 5e-3 * self.t_mid_tf) - 1.2))

        id = tf.math.less_equal(tf.abs(tf.log(self.Ex3)), 5.0)
        idx = tf.to_float(id, name='ToFloat')
        id2 = tf.math.less_equal(self.Ex2, 1e3)
        idx2 = tf.to_float(id2, name='ToFloat')

        self.L13_temp1 = self.D1 * self.u_x_mid_pred / self.k - (self.u_mid_pred * self.Ex1 - 1e2 * (1.0 - self.u_mid_pred) * self.Ex2)
        self.L13_temp2 = 1e2 * self.power_u * self.D1 * self.u_x_mid_pred / (self.Ex1 + self.power_v * self.Ex2) / self.k - 1e2 * 1e0 * self.u_mid_pred + 1e2 * self.power_v * self.Ex2 / (self.Ex1 + self.power_v * self.Ex2)
        self.L13 = self.L13_temp2

        self.L17_temp1 = self.u_x_mid_pred - self.k * (self.u_mid_pred_fin * self.Ex1 - 1e2 * (1.0 - self.u_mid_pred_fin) * self.Ex2) / self.D1
        self.L17_temp2 = 1e2 * self.D1 * self.u_x_mid_pred / (self.Ex1 + 1e2 * self.Ex2) / self.k - 1e2 * (self.u_mid_pred) + 1e2 * 1e2 * self.Ex2 / (self.Ex1 + 1e2 * self.Ex2)
        self.L17 = idx * self.L17_temp1 + (1 - idx) * self.L17_temp2

        self.loss1 = 20000 * tf.reduce_mean(tf.square(self.u0_pred - 1.0))
        self.loss5 = 20000 * tf.reduce_mean(tf.square(self.v0_pred - 0.0))
        self.loss2 = 10000 * tf.reduce_mean(tf.square(self.u_ub_pred - 1.0))
        self.loss6 = 10000 * tf.reduce_mean(tf.square(self.v_lb_pred - 0.0))
        self.loss3 = 3e3 * tf.reduce_mean(tf.square(1e1 * self.u_mid_pred - 1e1 * self.u0))
        self.loss4 = 1e2 * tf.reduce_mean(tf.square(self.f_u_pred))
        self.loss8 = 1e2 * tf.reduce_mean(tf.square(self.f_v_pred))
        self.loss13 = 1e1 * tf.reduce_mean(tf.square(self.L13))
        self.loss17 = 1e1 * tf.reduce_mean(tf.square(self.L17))

        self.loss = self.loss1 + self.loss2 + self.loss3 + self.loss4 + self.loss13
        self.Loss = self.loss17

        # Optimizers
        list1_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='former')
        list2_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='later')
        list3_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='temp')
        list_LBFGS = list1_vars + list2_vars + list3_vars
        list_LBFGS2 = list3_vars

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                var_list=list_LBFGS,
                                                                options={'maxiter': 7000,
                                                                         'maxfun': 20000,
                                                                         'maxcor': 70,
                                                                         'maxls': 70,
                                                                         'ftol': 0.01 * np.finfo(float).eps})

        self.optimizer2 = tf.contrib.opt.ScipyOptimizerInterface(self.Loss,
                                                                 method='L-BFGS-B',
                                                                 var_list=list_LBFGS2,
                                                                 options={'maxiter': 7000,
                                                                          'maxfun': 20000,
                                                                          'maxcor': 70,
                                                                          'maxls': 70,
                                                                          'ftol': 0.01 * np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate=0.0005)
        self.train_op1 = self.optimizer_Adam.minimize(self.loss, var_list=list1_vars + list2_vars + list3_vars)
        self.train_op2 = self.optimizer_Adam.minimize(self.Loss, var_list=list3_vars)

        # Session setup
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        init = tf.global_variables_initializer()
        self.sess.run(init)

    # === Helper Functions ===
    def initialize_NN(self, layers, countt):
        """
        Initialize weights and biases for a neural network.
        """
        weights = []
        biases = []
        num_layers = len(layers)

        for l in range(0, num_layers - 2):
            with tf.variable_scope("former"):
                W = self.xavier_init(size=[layers[l], layers[l + 1]])
                b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
                weights.append(W)
                biases.append(b)

        for l in range(num_layers - 2, num_layers - 1):
            with tf.variable_scope("later"):
                W = self.xavier_init(size=[layers[l], layers[l + 1]])
                b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
                weights.append(W)
                biases.append(b)

        return weights, biases

    def xavier_init(self, size):
        """
        Xavier initialization for weights.
        """
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    # === Core Network Functions ===
    def neural_net(self, X, weights_u, biases_u, weights_v, biases_v):
        """
        Define the neural network architecture for u and v.
        """
        num_layers = len(weights_u) + 1

        H = 100.0 * (X - self.lb_u) / (self.ub_u - self.lb_u) - 50.0
        for l in range(0, num_layers - 2):
            W = weights_u[l]
            b = biases_u[l]
            H = tf.nn.softplus(tf.add(tf.matmul(H, W), b))

        W = weights_u[-1]
        b = biases_u[-1]
        Y_u = tf.add(tf.matmul(H, W), b)

        H = 100.0 * (X - self.lb_v) / (self.ub_v - self.lb_v) - 50.0
        for l in range(0, num_layers - 2):
            W = weights_v[l]
            b = biases_v[l]
            H = tf.nn.softplus(tf.add(tf.matmul(H, W), b))

        W = weights_v[-1]
        b = biases_v[-1]
        Y_v = tf.add(tf.matmul(H, W), b)

        Y = tf.concat([Y_u, Y_v], axis=1)
        return Y, Y_u, Y_v

    def neural_net_D(self, u, u_x, weights_D, biases_D):
        """
        Define the neural network architecture for D.
        """
        num_layers = len(weights_D) + 1
        H = tf.concat([u, u_x], axis=1)

        for l in range(0, num_layers - 2):
            W = weights_D[l]
            b = biases_D[l]
            H = tf.nn.tanh(tf.add(tf.matmul(H, W), b))

        W = weights_D[-1]
        b = biases_D[-1]
        D = tf.add(tf.matmul(H, W), b)

        return D

    # === Prediction and Loss Functions ===
    def net_uv(self, x, t):
        """
        Compute u, v, and their gradients.
        """
        X = tf.concat([x, t], 1)
        uv, H_u, H_v = self.neural_net(X, self.weights_u, self.biases_u, self.weights_v, self.biases_v)
        u_temp = uv[:, 0:1]
        v_temp = uv[:, 1:2]
        u = u_temp
        v = v_temp

        u_x = tf.gradients(u, x)[0]
        v_x = tf.gradients(v, x)[0]

        return u, v, u_x, v_x

    def net_f_uv(self, x, t):
        """
        Compute the residual of the PDEs.
        """
        u, v, u_x, v_x = self.net_uv(x, t)
        u_t = tf.gradients(u, t)[0]
        u_xx = tf.gradients(u_x, x)[0]
        v_t = tf.gradients(v, t)[0]
        v_xx = tf.gradients(v_x, x)[0]

        D2_temp = self.neural_net_D(v, v_t, self.weights_D, self.biases_D)
        D2 = 1e-12 * tf.exp(D2_temp)

        D1_temp = self.neural_net_D(u, u_t, self.weights_u_D, self.biases_u_D)
        D1 = 1e-6 * tf.exp(D1_temp)

        f_u = 1e3 * u_t - 1e3 * 1e0 * self.D1 * u_xx
        f_v = 1e3 * v_t - 1e3 * 1e0 * self.D1 * v_xx

        return f_u, f_v, D1, D2

    # === Training and Prediction ===
    def callback(self, loss, loss1, loss2, loss13, loss4):
        """
        Print training progress.
        """
        print('Loss:', loss, loss1, loss2, loss13, loss4)

    def train(self, nIter):
        """
        Train the PINN model.
        """
        tf_dict = {self.x0_u_tf: self.x0_u, self.t0_u_tf: self.t0_u,
                   self.x0_v_tf: self.x0_v, self.t0_v_tf: self.t0_v,
                   self.x_lb_v_tf: self.x_lb_v, self.t_lb_v_tf: self.t_lb_v,
                   self.x_ub_u_tf: self.x_ub_u, self.t_ub_u_tf: self.t_ub_u,
                   self.x_mid_tf: self.x_mid, self.t_mid_tf: self.t_mid,
                   self.x_f_u_tf: self.x_f_u, self.t_f_u_tf: self.t_f_u,
                   self.x_f_v_tf: self.x_f_v, self.t_f_v_tf: self.t_f_v,
                   self.x_mid_end_tf: self.x_mid_end, self.t_mid_end_tf: self.t_mid_end}

        tf_dict2 = {self.x0_u_tf: self.x_mid_t_end_x, self.t0_u_tf: self.x_mid_t_end_t}



        for it in range(nIter):
            self.sess.run(self.train_op1, tf_dict)

            c_x_mid_t_end = self.sess.run(self.u0_pred, tf_dict2)
            c_x_x_mid_t_end = self.sess.run(self.u0_x_pred, tf_dict2)


            if it % 100 == 0:
                loss_value = self.sess.run(self.loss, tf_dict)
                loss_value1 = self.sess.run(self.loss1, tf_dict)
                loss_value2 = self.sess.run(self.loss2, tf_dict)
                loss_value13 = self.sess.run(self.loss13, tf_dict)
                loss_value4 = self.sess.run(self.loss4, tf_dict)

                print('It: %d, Loss: %.3e, loss1: %.3e, loss2: %.3e, loss13: %.3e, loss4: %.3e' %
                      (it, loss_value, loss_value1, loss_value2, loss_value13, loss_value4))

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss, self.loss1, self.loss2, self.loss13, self.loss4],
                                loss_callback=self.callback)


        self.save_path = self.saver.save(self.sess, '/results/model_new.ckpt')

        loss_value1 = self.sess.run(self.loss1, tf_dict)
        loss_value2 = self.sess.run(self.loss2, tf_dict)
        loss_value3 = self.sess.run(self.loss13, tf_dict)
        loss_value4 = self.sess.run(self.loss4, tf_dict)
        loss_value5 = self.sess.run(self.loss5, tf_dict)
        loss_value6 = self.sess.run(self.loss6, tf_dict)
        loss_value7 = self.sess.run(self.loss13, tf_dict)
        loss_value8 = self.sess.run(self.loss8, tf_dict)
        loss_value9 = self.sess.run(self.loss1, tf_dict)

        final_D1 = self.sess.run(self.power_u, tf_dict)
        final_k = self.sess.run(self.k, tf_dict)
        final_D2 = self.sess.run(self.power_v, tf_dict)
        final_E = self.sess.run(self.E, tf_dict)
        final_A = self.sess.run(self.alpha, tf_dict)

        print('Loss1: %.3e, Loss2: %.3e, Loss3: %.3e, Loss4: %.3e, Loss5: %.3e, Loss6: %.3e, Loss7: %.3e, Loss8: %.3e, Loss9: %.3e, D1: %.3e, k: %.3e , E: %.3e A: %.3e, D2:%.3e' %
              (loss_value1, loss_value2, loss_value3, loss_value4, loss_value5, loss_value6, loss_value7, loss_value8,
               loss_value9, final_D1, final_k, final_E, final_A, final_D2))

    def predict(self, X_u_star, X_v_star):
        """
        Make predictions using the trained PINN model.
        """
        tf_dict = {self.x_mid_tf: X_u_star[:, 0:1], self.t_mid_tf: X_u_star[:, 1:2]}
        u_star = self.sess.run(self.u_mid_pred, tf_dict)

        tf_dict = {self.x_mid_tf: X_v_star[:, 0:1], self.t_mid_tf: X_v_star[:, 1:2]}
        v_star = self.sess.run(self.v_mid_pred, tf_dict)

        tf_dict = {self.x0_u_tf: X_u_star[:, 0:1], self.t0_u_tf: X_u_star[:, 1:2]}
        f_u_star = self.sess.run(self.u0_x_pred, tf_dict)

        tf_dict = {self.x0_v_tf: X_v_star[:, 0:1], self.t0_v_tf: X_v_star[:, 1:2]}
        f_v_star = self.sess.run(self.v0_x_pred, tf_dict)

        tf_dict = {self.x_mid_tf: X_v_star[:, 0:1], self.t_mid_tf: X_v_star[:, 1:2]}
        D_star = self.sess.run(self.D2, tf_dict)

        tf_dict = {self.x_mid_tf: X_v_star[:, 0:1], self.t_mid_tf: X_v_star[:, 1:2]}
        D1_star = self.sess.run(self.D1, tf_dict)

        return u_star, v_star, f_u_star, f_v_star, D_star, D1_star