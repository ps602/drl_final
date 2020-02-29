import os
from agents.utils import *
from agents.policies import *
import logging
import multiprocessing as mp
import numpy as np
import tensorflow as tf

class DDPG:
    def __init__(self, n_s, n_a, total_step, model_config, seed=0, n_f=None):
        # load parameters
        self.name = 'ddpg'
        self.n_agent = 1
        # init reward norm/clip
        self.reward_clip = model_config.getfloat('reward_clip')
        self.reward_norm = model_config.getfloat('reward_norm')
        self.n_s = n_s
        self.n_a = n_a
        self.n_step = model_config.getint('batch_size')
        self.tau = model_config.getfloat('tau')  # TODO: create ini file for ddpg by copying a2c ini and add tau
        # init tf
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        self.policy = self._init_policy(n_s, n_a, n_f, model_config)
        self.saver = tf.train.Saver(max_to_keep=5)
        if total_step:
            # training
            self.total_step = total_step
            self._init_scheduler(model_config)
            self._init_train(model_config)
        self.sess.run(tf.global_variables_initializer())

    def _init_policy(self, n_s, n_a, n_w, n_f, model_config, agent_name=None):
        n_fw = model_config.getint('num_fw')
        n_ft = model_config.getint('num_ft')
        n_lstm = model_config.getint('num_lstm')
        if self.name == 'maddpg':
            n_fp = model_config.getint('num_fp')
            policy = FPLstmPGPolicy(n_s, n_a, n_w, n_f, self.n_step, n_fc_wave=n_fw,
                                    n_fc_wait=n_ft, n_fc_fp=n_fp, n_lstm=n_lstm, name=agent_name)
        else:
            policy = LstmPGPolicy(n_s, n_a, n_w, self.n_step, n_fc_wave=n_fw,
                                  n_fc_wait=n_ft, n_lstm=n_lstm, name=agent_name)
        return policy

    def _init_scheduler(self, model_config):  #TODO: change the metrics to that of MADDPG
        lr_init = model_config.getfloat('lr_init')
        lr_decay = model_config.get('lr_decay')
        #beta_init = model_config.getfloat('entropy_coef_init')
        #beta_decay = model_config.get('entropy_decay')
        if lr_decay == 'constant':
            self.lr_scheduler = Scheduler(lr_init, decay=lr_decay)
        else:
            lr_min = model_config.getfloat('LR_MIN')
            self.lr_scheduler = Scheduler(lr_init, lr_min, self.total_step, decay=lr_decay)
        #if beta_decay == 'constant':
        #    self.beta_scheduler = Scheduler(beta_init, decay=beta_decay)
        #else:
        #    beta_min = model_config.getfloat('ENTROPY_COEF_MIN')
        #    beta_ratio = model_config.getfloat('ENTROPY_RATIO')
        #    self.beta_scheduler = Scheduler(beta_init, beta_min, self.total_step * beta_ratio,
        #                                   decay=beta_decay)

    def _init_train(self, model_config):  #TODO: change the metrics to that of MADDPG
        # init loss
        #v_coef = model_config.getfloat('value_coef')
        max_grad_norm = model_config.getfloat('max_grad_norm')
        alpha = model_config.getfloat('rmsp_alpha')
        epsilon = model_config.getfloat('rmsp_epsilon')
        tau =  model_config.getfloat('tau')
        self.policy.prepare_loss(max_grad_norm, gamma, alpha, epsilon, tau)

        # init replay buffer
        buffer_size = model_config.getint('buffer_size')
        batch_size = model_config.getint('batch_size')
        self.trans_buffer = ReplayBuffer(buffer_size,batch_size)

    def save(self, model_dir, global_step):
        self.saver.save(self.sess, model_dir + 'checkpoint', global_step=global_step)

    def load(self, model_dir, checkpoint=None):
        save_file = None
        save_step = 0
        if os.path.exists(model_dir):
            if checkpoint is None:
                for file in os.listdir(model_dir):
                    if file.startswith('checkpoint'):
                        prefix = file.split('.')[0]
                        tokens = prefix.split('-')
                        if len(tokens) != 2:
                            continue
                        cur_step = int(tokens[1])
                        if cur_step > save_step:
                            save_file = prefix
                            save_step = cur_step
            else:
                save_file = 'checkpoint-' + str(int(checkpoint))
        if save_file is not None:
            self.saver.restore(self.sess, model_dir + save_file)
            logging.info('Checkpoint loaded: %s' % save_file)
            return True
        logging.error('Can not find old checkpoint for %s' % model_dir)
        return False

    def reset(self):
        self.policy._reset()

    def backward(self,out_type = 'pqvw',summary_writer=None, global_step=None):
        cur_lr = self.lr_scheduler.get(self.n_step)
        #cur_beta = self.beta_scheduler.get(self.n_step)
        #obs, acts, dones, Rs, Advs = self.trans_buffer.sample_transition(R)
        out_type = out_type
        obs, acts, nobs, Rs, dones= self.trans_buffer.sample_transition()#TODO: Chane samples coming from buffer
        self.policy.backward(self.sess, nobs, obs, acts, dones, Rs, cur_lr, out_type,
                             summary_writer=summary_writer, global_step=global_step)  #TODO: Change metrics

    def forward(self, ob, done, out_type='pqvw'):  #TODO: Change metrics
        return self.policy.forward(self.sess, ob, done, out_type)

    def add_transition(self, ob, next_obs, action, reward, done):  #TODO: Change acc to replay buffer necessities
        # Hard code the reward norm for negative reward only
        if (self.reward_norm):
            reward /= self.reward_norm
        if self.reward_clip:
            reward = np.clip(reward, -self.reward_clip, self.reward_clip)
        self.trans_buffer.add_transition(ob, next_obs, action, reward, done)


class IDDPG(DDPG):
    def __init__(self, n_s_ls, n_a_ls, n_w_ls, total_step,
                 model_config, seed=0):
        self.name = 'iddpg'
        self.agents = []
        self.n_agent = len(n_s_ls)  # length(n_s_ls) means number of agents
        self.reward_clip = model_config.getfloat('reward_clip')
        self.reward_norm = model_config.getfloat('reward_norm')
        self.n_s_ls = n_s_ls
        self.n_a_ls = n_a_ls
        self.n_w_ls = n_w_ls
        self.n_step = model_config.getint('batch_size')
        self.tau = model_config.getfloat('tau')# n_step is batch size
        self.buffer_size = model_config.getint('buffer_size')

        # init tf
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)  # ????????????
        self.policy_ls = []
        for i, (n_s, n_w, n_a) in enumerate(zip(self.n_s_ls, self.n_w_ls, self.n_a_ls)):
            # agent_name is needed to differentiate multi-agents
            self.policy_ls.append(self._init_policy(n_s - n_w, n_a, n_w, 0, model_config,
                                                    agent_name='{:d}a'.format(i)))
        self.saver = tf.train.Saver(max_to_keep=5)
        if total_step:
            # training
            self.total_step = total_step  # see beta
            self._init_scheduler(model_config)
            self._init_train(model_config)
        self.sess.run(tf.global_variables_initializer())

    def _init_train(self, model_config):  # TODO
        # init loss
        # v_coef = model_config.getfloat('value_coef')
        max_grad_norm = model_config.getfloat('max_grad_norm')
        alpha = model_config.getfloat('rmsp_alpha')
        epsilon = model_config.getfloat('rmsp_epsilon')
        gamma = model_config.getfloat('gamma')
        tau = model_config.getfloat('tau')
        coef = model_config.getfloat('coef')
        self.trans_buffer_ls = []
        for i in range(4):
            self.policy_ls[i].prepare_loss( max_grad_norm, gamma, alpha, epsilon, tau, coef)  # TODO
            self.trans_buffer_ls.append(ReplayBuffer(self.buffer_size,self.n_step))

    def backward_policy(self, y, obser, actios, nobser, Res, done, summary_writer=None, global_step=None,out_type = 'pqvw'):
        #print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%reachedmodelbackkwardpolicy')
        cur_lr = self.lr_scheduler.get(self.n_step)
        #cur_beta = self.beta_scheduler.get(self.n_step)
        #obs, acts, nobs, Rs, dones = self.trans_buffer_ls.sample_transition()_
        #print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%ymodeli')
        #print(y)
        for i in range(self.n_agent):
            #obs, acts, nobs, Rs, dones = self.trans_buffer_ls[i].sample_transition()
            obs = obser[i]
            acts = np.transpose(actios)
            nobs = nobser[i]
            Rs = Res[i]
            dones = done[i]
            x = y[i]
            val = self.policy_ls[i].v
            #print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%valuejesus')
            #print(val)
            # see once
            if i == 0:
                self.policy_ls[i].backward_policy(self.sess, val, x, nobs, obs, acts, dones, Rs, cur_lr,
                                          summary_writer=summary_writer, global_step=global_step)
            else:
                self.policy_ls[i].backward_policy(self.sess, val,  x, nobs, obs, acts, dones, Rs, cur_lr)

    def backward_value(self, y, obser, actios, nobser, Res, done, summary_writer=None, global_step=None,out_type = 'pqvw'):
        #print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%reachedmodelbackkwardpolicy')
        cur_lr = self.lr_scheduler.get(self.n_step)
        # cur_beta = self.beta_scheduler.get(self.n_step)
        # obs, acts, nobs, Rs, dones = self.trans_buffer_ls.sample_transition()_
        #print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%ymodeli')
        #print(y)
        for i in range(self.n_agent):
            # obs, acts, nobs, Rs, dones = self.trans_buffer_ls[i].sample_transition()
            obs = obser[i]
            acts = np.transpose(actios)
            nobs = nobser[i]
            Rs = Res[i]
            dones = done[i]
            x = y[i]
            val = self.policy_ls[i].v
            #print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%valuejesus')
            #print(val)
            # see once
            if i == 0:
                self.policy_ls[i].backward_value(self.sess, val, x, nobs, obs, acts, dones, Rs, cur_lr,
                                                  summary_writer=summary_writer, global_step=global_step)
            else:
                self.policy_ls[i].backward_value(self.sess, val, x, nobs, obs, acts, dones, Rs, cur_lr)
    def forward_value(self, obs, acs, done, out_type = 'pqvw' ):
        #print('++++++++++++++++++++++++++++++++reached model.forward')
        cur_out = []
        for i in range(self.n_agent):
            cur_out.append(self.policy_ls[i].forward_value(self.sess, acs, obs[i], done[i]))
        # cur_out_t = self.policy_ls[i].forward_t_policy(self.sess, obs[i], done)
        #print('++++++++++++++++++++++++++++++++++cur_outexecutes')


        return cur_out

    def forward_t_value(self, obs, acs, done, out_type = 'pqvw' ):
        #print('++++++++++++++++++++++++++++++++reached model.forward')
        cur_out = []
        for i in range(self.n_agent):
            cur_out.append(self.policy_ls[i].forward_target_value(self.sess, acs,obs[i], done[i]))
            # cur_out_t = self.policy_ls[i].forward_t_policy(self.sess, obs[i], done)
            #print('++++++++++++++++++++++++++++++++++cur_outexecutes')

            #print('+++++++++++++++++++++++++++cur_out')
            #print(cur_out)
        return cur_out
    def forward_t_policy(self, obs, done, out_type = 'pqvw' ):
        #print('++++++++++++++++++++++++++++++++reached model.forward')
        #print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%obsssmodel')
        #print(obs)
        #print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%donemodel')
        #print(done)
        cur_out = []
        for i in range(self.n_agent):
            cur_out.append(self.policy_ls[i].forward_t_policy(self.sess, obs[i], done[i]))
            #print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%cur_out forloop')
            #print(cur_out)
        # cur_out_t = self.policy_ls[i].forward_t_policy(self.sess, obs[i], done)
        #print('++++++++++++++++++++++++++++++++++cur_outexecutes')

        #print('+++++++++++++++++++++++++++cur_out')
        #print(cur_out)
        #if len(out_type) == 1:
        #    out.append(cur_out)
        #else:
        #    out1.append(cur_out[0])

        #if len(out_type) == 1:
        #    return out
        #else:
        #print('+++++++++++++++++++++++++++++++++++++++out1')
        #print(cur_out)
        return cur_out


    def forward_policy(self, obs, done, out_type='pqvw'): #obs 4,10 done 1

        cur_out =[]
        for i in range(self.n_agent):
            #print('=========================================obs[i]')
            #print(obs[i])
            # obs[i] 1,10
            # policy_ls is a list policy object
            cur_out.append(self.policy_ls[i].forward_policy(self.sess, obs[i], done))
        #cur_out_t = self.policy_ls[i].forward_t_policy(self.sess, obs[i], done)
        #print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%policyoutptmodels')
        #print(cur_out)  # returns 4 arrays of 2 elements each denoting prob of actions
        return cur_out

    def backward_mp(self,summary_writer=None, global_step=None):
        cur_lr = self.lr_scheduler.get(self.n_step)
        #cur_beta = self.beta_scheduler.get(self.n_step)

        def worker(i):
            obs, acts, nobs, Rs, dones = self.trans_buffer_ls[i].sample_transition()
            self.policy_ls[i].backward(self.sess, nobs, obs, acts, dones, Rs, cur_lr,
                                       summary_writer=summary_writer, global_step=global_step)

        mps = []
        for i in range(self.n_agent):
            p = mp.Process(target=worker, args=(i))
            p.start()
            mps.append(p)
        for p in mps:
            p.join()

    def reset(self):
        for policy in self.policy_ls:
            policy._reset()

    def add_transition(self, obs, actions, rewards, next_obs, done):
        if (self.reward_norm):
            rewards = np.array(rewards) / (self.reward_norm)
        if self.reward_clip:
            rewards = np.clip(rewards, -self.reward_clip, self.reward_clip)
        #print(actions)
        for i in range(self.n_agent):
            self.trans_buffer_ls[i].add_transition(obs[i], actions[i],
                                                   rewards[i], next_obs[i],done)
    def sample_transition(self):
        states = []
        actions = []
        next_states = []
        rewards = []
        dones = []
        for i in range(self.n_agent):

            state, action, next_state, reward, done = self.trans_buffer_ls[i].sample_transition()
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
            '''print('++++++++++++++++++++++++++++++++++++++++++++state')
            print(states)
            print('++++++++++++++++++++++++++++++++++++++++++++action')
            print(actions)
            print('++++++++++++++++++++++++++++++++++++++++++++next_state')
            print(next_states)
            print('++++++++++++++++++++++++++++++++++++++++++++reward')
            print(rewards)
            print('++++++++++++++++++++++++++++++++++++++++++++done')
            print(dones)'''


        return states, actions, next_states, rewards, dones







class MADDPG(IDDPG):
    def __init__(self, n_s_ls, n_a_ls, n_w_ls, n_f_ls, total_step,
                 model_config, seed=0):
        self.name = 'maddpg'
        self.agents = []
        self.n_agent = len(n_s_ls)
        self.reward_clip = model_config.getfloat('reward_clip')
        self.reward_norm = model_config.getfloat('reward_norm')
        self.n_s_ls = n_s_ls
        self.n_a_ls = n_a_ls
        self.n_f_ls = n_f_ls
        self.n_w_ls = n_w_ls
        self.n_step = model_config.getint('batch_size')
        self.buffer_size = model_config.getint('buffer_size')
        # init tf
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        config = tf.ConfigProto(allow_soft_placement=True)

        self.policy_ls = []
        for i, (n_s, n_a, n_w, n_f) in enumerate(zip(self.n_s_ls, self.n_a_ls, self.n_w_ls, self.n_f_ls)):
            # agent_name is needed to differentiate multi-agents
            
            self.policy_ls.append(self._init_policy(n_s - n_f - n_w, n_a, n_w, n_f, model_config,
                                                    agent_name='{:d}a'.format(i)))
        #print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%policy_ls')
        #print(self.policy_ls[2].v)
        self.sess = tf.Session(config=config)
        self.saver = tf.train.Saver(max_to_keep=5)
        if total_step:
            # training
            self.total_step = total_step
            self._init_scheduler(model_config)
            self._init_train(model_config)
        self.sess.run(tf.global_variables_initializer())



'''class A2C:
    def __init__(self, n_s, n_a, total_step, model_config, seed=0, n_f=None):
        # load parameters
        self.name = 'a2c'
        self.n_agent = 1
        # init reward norm/clip
        self.reward_clip = model_config.getfloat('reward_clip')
        self.reward_norm = model_config.getfloat('reward_norm')
        self.n_s = n_s
        self.n_a = n_a
        self.n_step = model_config.getint('batch_size')
        # init tf
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        self.policy = self._init_policy(n_s, n_a, n_f, model_config)
        self.saver = tf.train.Saver(max_to_keep=5)
        if total_step:
            # training
            self.total_step = total_step
            self._init_scheduler(model_config)
            self._init_train(model_config)
        self.sess.run(tf.global_variables_initializer())

    def _init_policy(self, n_s, n_a, n_w, n_f, model_config, agent_name=None):
        n_fw = model_config.getint('num_fw')
        n_ft = model_config.getint('num_ft')
        n_lstm = model_config.getint('num_lstm')
        if self.name == 'ma2c':
            n_fp = model_config.getint('num_fp')
            policy = FPLstmACPolicy(n_s, n_a, n_w, n_f, self.n_step, n_fc_wave=n_fw,
                                    n_fc_wait=n_ft, n_fc_fp=n_fp, n_lstm=n_lstm, name=agent_name)
        else:
            policy = LstmACPolicy(n_s, n_a, n_w, self.n_step, n_fc_wave=n_fw,
                                  n_fc_wait=n_ft, n_lstm=n_lstm, name=agent_name)
        return policy

    def _init_scheduler(self, model_config):
        lr_init = model_config.getfloat('lr_init')
        lr_decay = model_config.get('lr_decay')
        beta_init = model_config.getfloat('entropy_coef_init')
        beta_decay = model_config.get('entropy_decay')
        if lr_decay == 'constant':
            self.lr_scheduler = Scheduler(lr_init, decay=lr_decay)
        else:
            lr_min = model_config.getfloat('LR_MIN')
            self.lr_scheduler = Scheduler(lr_init, lr_min, self.total_step, decay=lr_decay)
        if beta_decay == 'constant':
            self.beta_scheduler = Scheduler(beta_init, decay=beta_decay)
        else:
            beta_min = model_config.getfloat('ENTROPY_COEF_MIN')
            beta_ratio = model_config.getfloat('ENTROPY_RATIO')
            self.beta_scheduler = Scheduler(beta_init, beta_min, self.total_step * beta_ratio,
                                            decay=beta_decay)

    def _init_train(self, model_config):
        # init loss
        v_coef = model_config.getfloat('value_coef')
        max_grad_norm = model_config.getfloat('max_grad_norm')
        alpha = model_config.getfloat('rmsp_alpha')
        epsilon = model_config.getfloat('rmsp_epsilon')
        self.policy.prepare_loss(v_coef, max_grad_norm, alpha, epsilon)

        # init replay buffer
        gamma = model_config.getfloat('gamma')#TODO: Replay buffer?
        self.trans_buffer = OnPolicyBuffer(gamma)

    def save(self, model_dir, global_step):
        self.saver.save(self.sess, model_dir + 'checkpoint', global_step=global_step)

    def load(self, model_dir, checkpoint=None):
        save_file = None
        save_step = 0
        if os.path.exists(model_dir):
            if checkpoint is None:
                for file in os.listdir(model_dir):
                    if file.startswith('checkpoint'):
                        prefix = file.split('.')[0]
                        tokens = prefix.split('-')
                        if len(tokens) != 2:
                            continue
                        cur_step = int(tokens[1])
                        if cur_step > save_step:
                            save_file = prefix
                            save_step = cur_step
            else:
                save_file = 'checkpoint-' + str(int(checkpoint))
        if save_file is not None:
            self.saver.restore(self.sess, model_dir + save_file)
            logging.info('Checkpoint loaded: %s' % save_file)
            return True
        logging.error('Can not find old checkpoint for %s' % model_dir)
        return False

    def reset(self):
        self.policy._reset()

    def backward(self, R, summary_writer=None, global_step=None):
        cur_lr = self.lr_scheduler.get(self.n_step)
        cur_beta = self.beta_scheduler.get(self.n_step)
        obs, acts, dones, Rs, Advs = self.trans_buffer.sample_transition(R)
        self.policy.backward(self.sess, obs, acts, dones, Rs, Advs, cur_lr, cur_beta,
                             summary_writer=summary_writer, global_step=global_step)

    def forward(self, nob, ob, done, out_type='pv'):
        return self.policy.forward(self.sess, nob, ob, done, out_type)

    def add_transition(self, ob, action, reward, value, done):
        # Hard code the reward norm for negative reward only
        if (self.reward_norm):
            reward /= self.reward_norm
        if self.reward_clip:
            reward = np.clip(reward, -self.reward_clip, self.reward_clip)
        self.trans_buffer.add_transition(ob, action, reward, value, done)



class IA2C(A2C):
    def __init__(self, n_s_ls, n_a_ls, n_w_ls, total_step,
                 model_config, seed=0):
        self.name = 'ia2c'
        self.agents = []
        self.n_agent = len(n_s_ls)  # length(n_s_ls) means number of agents
        self.reward_clip = model_config.getfloat('reward_clip')
        self.reward_norm = model_config.getfloat('reward_norm')
        self.n_s_ls = n_s_ls
        self.n_a_ls = n_a_ls
        self.n_w_ls = n_w_ls
        self.n_step = model_config.getint('batch_size')  # n_step is batch size
        # init tf
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)  # ????????????
        self.policy_ls = []
        for i, (n_s, n_w, n_a) in enumerate(zip(self.n_s_ls, self.n_w_ls, self.n_a_ls)):
            # agent_name is needed to differentiate multi-agents
            self.policy_ls.append(self._init_policy(n_s - n_w, n_a, n_w, 0, model_config,
                                  agent_name='{:d}a'.format(i)))
        self.saver = tf.train.Saver(max_to_keep=5)
        if total_step:
            # training
            self.total_step = total_step  # see beta
            self._init_scheduler(model_config)
            self._init_train(model_config)
        self.sess.run(tf.global_variables_initializer())

    def _init_train(self, model_config):#TODO
        # init loss
        v_coef = model_config.getfloat('value_coef')
        max_grad_norm = model_config.getfloat('max_grad_norm')
        alpha = model_config.getfloat('rmsp_alpha')
        epsilon = model_config.getfloat('rmsp_epsilon')
        gamma = model_config.getfloat('gamma')
        self.trans_buffer_ls = []
        for i in range(self.n_agent):
            self.policy_ls[i].prepare_loss(v_coef, max_grad_norm, alpha, epsilon)#TODO
            self.trans_buffer_ls.append(OnPolicyBuffer(gamma))

    def backward(self, R_ls, summary_writer=None, global_step=None):
        cur_lr = self.lr_scheduler.get(self.n_step)
        cur_beta = self.beta_scheduler.get(self.n_step)
        for i in range(self.n_agent):
            obs, acts, dones, Rs, Advs = self.trans_buffer_ls[i].sample_transition(R_ls[i])  # see once
            if i == 0:
                self.policy_ls[i].backward(self.sess, obs, acts, dones, Rs, Advs, cur_lr, cur_beta,
                                           summary_writer=summary_writer, global_step=global_step)
            else:
                self.policy_ls[i].backward(self.sess, obs, acts, dones, Rs, Advs, cur_lr, cur_beta)

    def forward(self, nob, obs, done, out_type='pv'):
        if len(out_type) == 1:
            out = []
        elif len(out_type) == 2:
            out1, out2 = [], []
        for i in range(self.n_agent):
            cur_out = self.policy_ls[i].forward(self.sess, nob[i], obs[i], done, out_type)
            if len(out_type) == 1:
                out.append(cur_out)
            else:
                out1.append(cur_out[0])
                out2.append(cur_out[1])
        if len(out_type) == 1:
            return out
        else:
            return out1, out2

    def backward_mp(self, R_ls, summary_writer=None, global_step=None):
        cur_lr = self.lr_scheduler.get(self.n_step)
        cur_beta = self.beta_scheduler.get(self.n_step)

        def worker(i):
            obs, acts, dones, Rs, Advs = self.trans_buffer_ls[i].sample_transition(R_ls[i])
            self.policy_ls[i].backward(self.sess, obs, acts, dones, Rs, Advs, cur_lr, cur_beta,
                                       summary_writer=summary_writer, global_step=global_step)
        mps = []
        for i in range(self.n_agent):
            p = mp.Process(target=worker, args=(i))
            p.start()
            mps.append(p)
        for p in mps:
            p.join()

    def reset(self):
        for policy in self.policy_ls:
            policy._reset()

    def add_transition(self, obs, actions, rewards, values, done):
        if (self.reward_norm):
            rewards = rewards / self.reward_norm
        if self.reward_clip:
            rewards = np.clip(rewards, -self.reward_clip, self.reward_clip)
        for i in range(self.n_agent):
            self.trans_buffer_ls[i].add_transition(obs[i], actions[i],
                                                   rewards[i], values[i], done)


class MA2C(IA2C):
    def __init__(self, n_s_ls, n_a_ls, n_w_ls, n_f_ls, total_step,
                 model_config, seed=0):
        self.name = 'ma2c'
        self.agents = []
        self.n_agent = len(n_s_ls)
        self.reward_clip = model_config.getfloat('reward_clip')
        self.reward_norm = model_config.getfloat('reward_norm')
        self.n_s_ls = n_s_ls
        self.n_a_ls = n_a_ls
        self.n_f_ls = n_f_ls
        self.n_w_ls = n_w_ls
        self.n_step = model_config.getint('batch_size')
        # init tf
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        self.policy_ls = []
        for i, (n_s, n_a, n_w, n_f) in enumerate(zip(self.n_s_ls, self.n_a_ls, self.n_w_ls, self.n_f_ls)):
            # agent_name is needed to differentiate multi-agents
            self.policy_ls.append(self._init_policy(n_s - n_f - n_w, n_a, n_w, n_f, model_config,
                                                    agent_name='{:d}a'.format(i)))
        self.saver = tf.train.Saver(max_to_keep=5)
        if total_step:
            # training
            self.total_step = total_step
            self._init_scheduler(model_config)
            self._init_train(model_config)
        self.sess.run(tf.global_variables_initializer())


class IQL(A2C):
    def __init__(self, n_s_ls, n_a_ls, n_w_ls, total_step, model_config, seed=0, model_type='dqn'):
        self.name = 'iql'
        self.model_type = model_type
        self.agents = []
        self.n_agent = len(n_s_ls)
        self.reward_clip = model_config.getfloat('reward_clip')
        self.reward_norm = model_config.getfloat('reward_norm')
        self.n_s_ls = n_s_ls
        self.n_a_ls = n_a_ls
        self.n_w_ls = n_w_ls
        self.n_step = model_config.getint('batch_size')
        # init tf
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        self.policy_ls = []
        for i, (n_s, n_a, n_w) in enumerate(zip(self.n_s_ls, self.n_a_ls, self.n_w_ls)):
            # agent_name is needed to differentiate multi-agents
            self.policy_ls.append(self._init_policy(n_s, n_a, n_w, model_config,
                                                    agent_name='{:d}a'.format(i)))
        self.saver = tf.train.Saver(max_to_keep=5)
        if total_step:
            # training
            self.total_step = total_step
            self._init_scheduler(model_config)
            self._init_train(model_config)
        self.cur_step = 0
        self.sess.run(tf.global_variables_initializer())

    def _init_policy(self, n_s, n_a, n_w, model_config, agent_name=None):
        if self.model_type == 'dqn':
            n_h = model_config.getint('num_h')
            n_fc = model_config.getint('num_fc')
            policy = DeepQPolicy(n_s - n_w, n_a, n_w, self.n_step, n_fc0=n_fc, n_fc=n_h,
                                 name=agent_name)
        else:
            policy = LRQPolicy(n_s, n_a, self.n_step, name=agent_name)
        return policy

    def _init_scheduler(self, model_config):
        lr_init = model_config.getfloat('lr_init')
        lr_decay = model_config.get('lr_decay')
        eps_init = model_config.getfloat('epsilon_init')
        eps_decay = model_config.get('epsilon_decay')
        if lr_decay == 'constant':
            self.lr_scheduler = Scheduler(lr_init, decay=lr_decay)
        else:
            lr_min = model_config.getfloat('lr_min')
            self.lr_scheduler = Scheduler(lr_init, lr_min, self.total_step, decay=lr_decay)
        if eps_decay == 'constant':
            self.eps_scheduler = Scheduler(eps_init, decay=eps_decay)
        else:
            eps_min = model_config.getfloat('epsilon_min')
            eps_ratio = model_config.getfloat('epsilon_ratio')
            self.eps_scheduler = Scheduler(eps_init, eps_min, self.total_step * eps_ratio,
                                           decay=eps_decay)

    def _init_train(self, model_config):
        # init loss
        max_grad_norm = model_config.getfloat('max_grad_norm')
        gamma = model_config.getfloat('gamma')
        buffer_size = model_config.getfloat('buffer_size')
        self.trans_buffer_ls = []
        for i in range(self.n_agent):
            self.policy_ls[i].prepare_loss(max_grad_norm, gamma)
            self.trans_buffer_ls.append(ReplayBuffer(buffer_size, self.n_step))

    def backward(self, summary_writer=None, global_step=None):
        cur_lr = self.lr_scheduler.get(self.n_step)
        if self.trans_buffer_ls[0].size < self.trans_buffer_ls[0].batch_size:
            return
        for i in range(self.n_agent):
            for k in range(10):
                obs, acts, next_obs, rs, dones = self.trans_buffer_ls[i].sample_transition()
                if i == 0:
                    self.policy_ls[i].backward(self.sess, obs, acts, next_obs, dones, rs, cur_lr,
                                               summary_writer=summary_writer,
                                               global_step=global_step + k)
                else:
                    self.policy_ls[i].backward(self.sess, obs, acts, next_obs, dones, rs, cur_lr)

    def forward(self, obs, mode='act'):
        if mode == 'explore':
            eps = self.eps_scheduler.get(1)
        action = []
        qs_ls = []
        for i in range(self.n_agent):
            qs = self.policy_ls[i].forward(self.sess, obs[i])
            if (mode == 'explore') and (np.random.random() < eps):
                action.append(np.random.randint(self.n_a_ls[i]))
            else:
                action.append(np.argmax(qs))
            qs_ls.append(qs)
        return action, qs_ls

    def reset(self):
        # do nothing
        return

    def add_transition(self, obs, actions, rewards, next_obs, done):
        if (self.reward_norm):
            rewards = rewards / self.reward_norm
        if self.reward_clip:
            rewards = np.clip(rewards, -self.reward_clip, self.reward_clip)
        for i in range(self.n_agent):
            self.trans_buffer_ls[i].add_transition(obs[i], actions[i],
                                                   rewards[i], next_obs[i], done)'''
