import numpy as np
import tensorflow as tf
from agents.utils import *
import bisect



class ACPolicy:
    def __init__(self, n_a, n_s, n_step, policy_name, agent_name):
        self.name = policy_name  # name of policy
        if agent_name is not None:
            # for multi-agent system
            self.name += '_' + str(agent_name)  # appending policy_name_a2c ot policy_name_ma2c
        self.n_a = n_a  # number of actions
        self.n_s = n_s  # number of states
        self.n_step = n_step  # number of steps

    def forward(self, ob, *_args, **_kwargs):
        raise NotImplementedError()

    def _build_out_net(self, h, out_type):
        if out_type == 'pi':  # if out_type is pi (??)
            pi = fc(h, out_type, self.n_a, act=tf.nn.softmax)  # calling fc function and passing h as x, out_type as scope, number of actions as n_out and act as softmax
            return tf.squeeze(pi)  # removes items of dimension 1 from tensor of many dimensions
        else:
            v = fc(h, out_type, 1, act=lambda x: x)  # step function as act and n_out as 1 when out_type is not pi
            return tf.squeeze(v)  # removes items of dimension 1 from tensor of many dimensions

# one forward pass

    def _get_forward_outs(self, out_type):
        outs = []
        if 'p' in out_type:  # check out_type is 'p' or out_type is 'v' and correspondingly append v or pi to outs
            outs.append(self.pi)
        if 'v' in out_type:
            outs.append(self.v)
        return outs

# appending forward outputs

    def _return_forward_outs(self, out_values):
        if len(out_values) == 1:
            return out_values[0]
        return out_values



    def prepare_loss(self, v_coef, max_grad_norm, alpha, epsilon):
        self.A = tf.placeholder(tf.int32, [self.n_step])  # just creates a skeletal value for variable A that's not declared yet
        self.ADV = tf.placeholder(tf.float32, [self.n_step])  # just creates a skeletal value for variable ADV that's not declared yet
        self.R = tf.placeholder(tf.float32, [self.n_step])  # just creates a skeletal value for variable R that's not declared yet
        self.entropy_coef = tf.placeholder(tf.float32, [])  # just creates a skeletal value for variable entropy_coef that's not declared yet
        A_sparse = tf.one_hot(self.A, self.n_a)
        # one hot takes in indices (self.A) and depth (n_a), where it o/p a matrix of col size n_A and row size as size of self.A
        # the values of self.A are the indices of the row that have on_value
        log_pi = tf.log(tf.clip_by_value(self.pi, 1e-10, 1.0))  # clipping value of pi between 1e-10 and 1 and taking log of it
        entropy = -tf.reduce_sum(self.pi * log_pi, axis=1)  # defining entropy as -sum along rows of self.pi and log_pi
        entropy_loss = -tf.reduce_mean(entropy) * self.entropy_coef  # defining entropy_loss as mean of entropy * entropy_coef
        policy_loss = -tf.reduce_mean(tf.reduce_sum(log_pi * A_sparse, axis=1) * self.ADV)  # ??
        value_loss = tf.reduce_mean(tf.square(self.R - self.v)) * 0.5 * v_coef  # ??
        self.loss = policy_loss + value_loss + entropy_loss  # ??

        wts = tf.trainable_variables(scope=self.name)  # within the scope of the policy name we declare trainable variables
        grads = tf.gradients(self.loss, wts)  # gradient of loss wrt wts
        if max_grad_norm > 0:
            grads, self.grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)  # grad clipping
        self.lr = tf.placeholder(tf.float32, [])  # learning rate
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=alpha, epsilon=epsilon)  # declaring optimiser as RMSPropOptimiser lr,beta = alpha and epsilon
        self._train = self.optimizer.apply_gradients(list(zip(grads, wts)))  # arg needs to be a list containing grads and weights (variables)
        # monitor training
        if self.name.endswith('_0a'):
            summaries = []
            # summaries.append(tf.summary.scalar('loss/%s_entropy_loss' % self.name, entropy_loss))
            summaries.append(tf.summary.scalar('loss/%s_policy_loss' % self.name, policy_loss))
            summaries.append(tf.summary.scalar('loss/%s_value_loss' % self.name, value_loss))
            summaries.append(tf.summary.scalar('loss/%s_total_loss' % self.name, self.loss))
            # summaries.append(tf.summary.scalar('train/%s_lr' % self.name, self.lr))
            # summaries.append(tf.summary.scalar('train/%s_entropy_beta' % self.name, self.entropy_coef))
            summaries.append(tf.summary.scalar('train/%s_gradnorm' % self.name, self.grad_norm))
            self.summary = tf.summary.merge(summaries)


class LstmACPolicy(ACPolicy):
    def __init__(self, n_s, n_a, n_w, n_step, n_fc_wave=128, n_fc_wait=32, n_lstm=64, name=None):
        super().__init__(n_a, n_s, n_step, 'lstm', name)
        self.n_lstm = n_lstm  # n_s, n_w, n_a??
        self.n_fc_wait = n_fc_wait  # number of fc_wait
        self.n_fc_wave = n_fc_wave  # number of fc_wave
        self.n_w = n_w  # n_w
        self.ob_fw = tf.placeholder(tf.float32, [1, n_s + n_w])  # forward 1-step are these observations?
        self.done_fw = tf.placeholder(tf.float32, [1])  # are these done?
        self.ob_bw = tf.placeholder(tf.float32, [n_step, n_s + n_w])  # backward n-step
        self.done_bw = tf.placeholder(tf.float32, [n_step])
        self.states = tf.placeholder(tf.float32, [2, n_lstm * 2])  # defining states with the size of (2,n_lstm*2)
        with tf.variable_scope(self.name):
            # pi and v use separate nets
            self.pi_fw, pi_state = self._build_net('forward', 'pi')  # calling _build_net function for both the networks pi & v
            self.v_fw, v_state = self._build_net('forward', 'v')
            pi_state = tf.expand_dims(pi_state, 0)
            v_state = tf.expand_dims(v_state, 0)
            self.new_states = tf.concat([pi_state, v_state], 0)
        with tf.variable_scope(self.name, reuse=True):
            self.pi, _ = self._build_net('backward', 'pi')
            self.v, _ = self._build_net('backward', 'v')
        self._reset()

    def _build_net(self, in_type, out_type):  # defining _build_net with in_type and out_type as arguments
        if in_type == 'forward':  # if the in_type is forward
            ob = self.ob_fw  # setting ob as ob_fw
            done = self.done_fw  # setting done as done_fw
        else:  # if the in_type is backward
            ob = self.ob_bw   # setting ob as ob_bw
            done = self.done_bw  # setting done as done_bw
        if out_type == 'pi':  # if out_type is pi (policy network)
            states = self.states[0]  # setting states as states[0] for policy network
        else:  # if out_type is v (value network)
            states = self.states[1]   # setting states as states[1] for value network
        if self.n_w == 0:  # if n_w is 0
            h = fc(ob, out_type + '_fcw', self.n_fc_wave)  # calling fc function
        else:
            h0 = fc(ob[:, :self.n_s], out_type + '_fcw', self.n_fc_wave)
            h1 = fc(ob[:, self.n_s:], out_type + '_fct', self.n_fc_wait)
            h = tf.concat([h0, h1], 1)
        h, new_states = lstm(h, done, states, out_type + '_lstm')
        out_val = self._build_out_net(h, out_type)
        return out_val, new_states

    def _reset(self):
        # forget the cumulative states every cum_step
        self.states_fw = np.zeros((2, self.n_lstm * 2), dtype=np.float32)
        self.states_bw = np.zeros((2, self.n_lstm * 2), dtype=np.float32)

    def forward(self, sess, ob, done, out_type='pv'):
        outs = self._get_forward_outs(out_type)
        # update state only when p is called
        if 'p' in out_type:
            outs.append(self.new_states)
        out_values = sess.run(outs, {self.ob_fw:np.array([ob]),
                                     self.done_fw:np.array([done]),
                                     self.states:self.states_fw})
        if 'p' in out_type:
            self.states_fw = out_values[-1]
            out_values = out_values[:-1]
        return self._return_forward_outs(out_values)

    def backward(self, sess, obs, acts, dones, Rs, Advs, cur_lr, cur_beta,
                 summary_writer=None, global_step=None):
        if summary_writer is None:
            ops = self._train
        else:
            ops = [self.summary, self._train]
        outs = sess.run(ops,
                        {self.ob_bw: obs,
                         self.done_bw: dones,
                         self.states: self.states_bw,
                         self.A: acts,
                         self.ADV: Advs,
                         self.R: Rs,
                         self.lr: cur_lr,
                         self.entropy_coef: cur_beta})
        self.states_bw = np.copy(self.states_fw)
        if summary_writer is not None:
            summary_writer.add_summary(outs[0], global_step=global_step)

    def _get_forward_outs(self, out_type):
        outs = []
        if 'p' in out_type:
            outs.append(self.pi_fw)
        if 'v' in out_type:
            outs.append(self.v_fw)
        return outs


class FPLstmACPolicy(LstmACPolicy):
    def __init__(self, n_s, n_a, n_w, n_f, n_step, n_fc_wave=128, n_fc_wait=32, n_fc_fp=32, n_lstm=64, name=None):
        ACPolicy.__init__(self, n_a, n_s, n_step, 'fplstm', name)
        self.n_lstm = n_lstm
        self.n_fc_wave = n_fc_wave
        self.n_fc_wait = n_fc_wait
        self.n_fc_fp = n_fc_fp
        self.n_w = n_w
        self.ob_fw = tf.placeholder(tf.float32, [1, n_s + n_w + n_f]) # forward 1-step
        self.done_fw = tf.placeholder(tf.float32, [1])
        self.ob_bw = tf.placeholder(tf.float32, [n_step, n_s + n_w + n_f]) # backward n-step
        self.done_bw = tf.placeholder(tf.float32, [n_step])
        self.states = tf.placeholder(tf.float32, [2, n_lstm * 2])
        with tf.variable_scope(self.name):
            # pi and v use separate nets
            self.pi_fw, pi_state = self._build_net('forward', 'pi')
            self.v_fw, v_state = self._build_net('forward', 'v')
            pi_state = tf.expand_dims(pi_state, 0)
            v_state = tf.expand_dims(v_state, 0)
            self.new_states = tf.concat([pi_state, v_state], 0)
        with tf.variable_scope(self.name, reuse=True):
            self.pi, _ = self._build_net('backward', 'pi')
            self.v, _ = self._build_net('backward', 'v')
        self._reset()

    def _build_net(self, in_type, out_type):
        if in_type == 'forward':
            ob = self.ob_fw
            done = self.done_fw
        else:
            ob = self.ob_bw
            done = self.done_bw
        if out_type == 'pi':
            states = self.states[0]
        else:
            states = self.states[1]
        h0 = fc(ob[:, :self.n_s], out_type + '_fcw', self.n_fc_wave)
        h1 = fc(ob[:, (self.n_s + self.n_w):], out_type + '_fcf', self.n_fc_fp)
        if self.n_w == 0:
            h = tf.concat([h0, h1], 1)
        else:
            h2 = fc(ob[:, self.n_s: (self.n_s + self.n_w)], out_type + '_fct', self.n_fc_wait)
            h = tf.concat([h0, h1, h2], 1)
        h, new_states = lstm(h, done, states, out_type + '_lstm')
        out_val = self._build_out_net(h, out_type)
        return out_val, new_states


class FcACPolicy(ACPolicy):
    def __init__(self, n_s, n_a, n_w, n_step, n_fc_wave=128, n_fc_wait=32, n_lstm=64, name=None):
        super().__init__(n_a, n_s, n_step, 'fc', name)
        self.n_fc_wave = n_fc_wave
        self.n_fc_wait = n_fc_wait
        self.n_fc = n_lstm
        self.n_w = n_w
        self.obs = tf.placeholder(tf.float32, [None, n_s + n_w])
        with tf.variable_scope(self.name):
            # pi and v use separate nets
            self.pi = self._build_net('pi')
            self.v = self._build_net('v')

    def _build_net(self, out_type):
        if self.n_w == 0:
            h = fc(self.obs, out_type + '_fcw', self.n_fc_wave)
        else:
            h0 = fc(self.obs[:, :self.n_s], out_type + '_fcw', self.n_fc_wave)
            h1 = fc(self.obs[:, self.n_s:], out_type + '_fct', self.n_fc_wait)
            h = tf.concat([h0, h1], 1)
        h = fc(h, out_type + '_fc', self.n_fc)
        return self._build_out_net(h, out_type)

    def forward(self, sess, ob, done, out_type='pv'):
        outs = self._get_forward_outs(out_type)
        out_values = sess.run(outs, {self.obs: np.array([ob])})
        return self._return_forward_outs(out_values)

    def backward(self, sess, obs, acts, dones, Rs, Advs, cur_lr, cur_beta,
                 summary_writer=None, global_step=None):
        if summary_writer is None:
            ops = self._train
        else:
            ops = [self.summary, self._train]
        outs = sess.run(ops,
                        {self.obs: obs,
                         self.A: acts,
                         self.ADV: Advs,
                         self.R: Rs,
                         self.lr: cur_lr,
                         self.entropy_coef: cur_beta})
        if summary_writer is not None:
            summary_writer.add_summary(outs[0], global_step=global_step)


class FPFcACPolicy(FcACPolicy):
    def __init__(self, n_s, n_a, n_w, n_f, n_step, n_fc_wave=128, n_fc_wait=32, n_fc_fp=32, n_lstm=64, name=None):
        ACPolicy.__init__(self, n_a, n_s, n_step, 'fpfc', name)
        self.n_fc_wave = n_fc_wave
        self.n_fc_wait = n_fc_wait
        self.n_fc_fp = n_fc_fp
        self.n_fc = n_lstm
        self.n_w = n_w
        self.obs = tf.placeholder(tf.float32, [None, n_s + n_w + n_f])
        with tf.variable_scope(self.name):
            # pi and v use separate nets
            self.pi = self._build_net('pi')
            self.v = self._build_net('v')

    def _build_net(self, out_type):
        h0 = fc(ob[:, :self.n_s], out_type + '_fcw', self.n_fc_wave)
        h1 = fc(ob[:, (self.n_s + self.n_w):], out_type + '_fcf', self.n_fc_fp)
        if self.n_w == 0:
            h = tf.concat([h0, h1], 1)
        else:
            h2 = fc(ob[:, self.n_s: (self.n_s + self.n_w)], out_type + '_fct', self.n_fc_wait)
            h = tf.concat([h0, h1, h2], 1)
        h = fc(h, out_type + '_fc', self.n_fc)
        return self._build_out_net(h, out_type)

