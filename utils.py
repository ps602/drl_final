import itertools
import logging
import numpy as np
import tensorflow as tf
import time
import os
import pandas as pd
import subprocess

def sample_gumbel(shape, eps=1e-20):
  """Sample from Gumbel(0, 1)"""
  U = tf.random_uniform(shape,minval=0,maxval=1)
  return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
  """ Draw a sample from the Gumbel-Softmax distribution"""
  y = logits + sample_gumbel(tf.shape(logits))
  return tf.nn.softmax( y / temperature)

def gumbel_softmax(logits, temperature = 1, hard=False):
  """Sample from the Gumbel-Softmax distribution and optionally discretize.
  Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
  Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
  """
  y = gumbel_softmax_sample(logits, temperature)
  if hard:
    k = tf.shape(logits)[-1]
    #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
    y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,keep_dims=True)),y.dtype)
    y = tf.stop_gradient(y_hard - y) + y
  return y



def check_dir(cur_dir):
    if not os.path.exists(cur_dir):
        return False
    return True


def copy_file(src_dir, tar_dir):
    cmd = 'cp %s %s' % (src_dir, tar_dir)
    subprocess.check_call(cmd, shell=True)


def find_file(cur_dir, suffix='.ini'):
    for file in os.listdir(cur_dir):
        if file.endswith(suffix):
            return cur_dir + '/' + file
    logging.error('Cannot find %s file' % suffix)
    return None


def init_dir(base_dir, pathes=['log', 'data', 'model']):
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    dirs = {}
    for path in pathes:
        cur_dir = base_dir + '/%s/' % path
        if not os.path.exists(cur_dir):
            os.mkdir(cur_dir)
        dirs[path] = cur_dir
    return dirs


def init_log(log_dir):
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                        level=logging.INFO,
                        handlers=[
                            logging.FileHandler('%s/%d.log' % (log_dir, time.time())),
                            logging.StreamHandler()
                        ])


def init_test_flag(test_mode):
    if test_mode == 'no_test':
        return False, False
    if test_mode == 'in_train_test':
        return True, False
    if test_mode == 'after_train_test':
        return False, True
    if test_mode == 'all_test':
        return True, True
    return False, False


def plot_train(data_dirs, labels):
    pass

def plot_evaluation(data_dirs, labels):
    pass


class Counter:
    def __init__(self, total_step, test_step, log_step):
        self.counter = itertools.count(1)
        self.cur_step = 0
        self.cur_test_step = 0
        self.total_step = total_step
        self.test_step = test_step
        self.log_step = log_step
        self.stop = False
        # self.init_test = True

    def next(self):
        self.cur_step = next(self.counter)
        return self.cur_step

    def should_test(self):
        # if self.init_test:
        #     self.init_test = False
        #     return True
        test = False
        if (self.cur_step - self.cur_test_step) >= self.test_step:
            test = True
            self.cur_test_step = self.cur_step
        return test

    # def update_test(self, reward):
    #     if self.prev_reward is not None:
    #         if abs(self.prev_reward - reward) <= self.delta_reward:
    #             self.stop = True
    #     self.prev_reward = reward

    def should_log(self):
        return (self.cur_step % self.log_step == 0)

    def should_stop(self):
        if self.cur_step >= self.total_step:
            return True
        return self.stop


class Trainer():
    def __init__(self, env, model, global_counter, summary_writer, run_test, output_path=None):
        self.cur_step = 0
        self.global_counter = global_counter
        self.env = env
        self.agent = self.env.agent
        self.model = model
        self.sess = self.model.sess
        self.n_step = self.model.n_step
        self.summary_writer = summary_writer
        self.run_test = run_test
        assert self.env.T % self.n_step == 0
        self.data = []
        self.output_path = output_path
        if run_test:
            self.test_num = self.env.test_num
            logging.info('Testing: total test num: %d' % self.test_num)
        self._init_summary()

    def _init_summary(self):
        self.train_reward = tf.placeholder(tf.float32, [])
        self.train_summary = tf.summary.scalar('train_reward', self.train_reward)
        #self.test_reward = tf.placeholder(tf.float32, [])
        #self.test_summary = tf.summary.scalar('test_reward', self.test_reward)

    def _add_summary(self, reward, global_step, is_train=True):
        if is_train:
            summ = self.sess.run(self.train_summary, {self.train_reward: reward})
        else:
            summ = self.sess.run(self.test_summary, {self.test_reward: reward})
        self.summary_writer.add_summary(summ, global_step=global_step)

    def explore(self, prev_ob, prev_done):
        ob = prev_ob
        #print('+++============================================++ob')
        #print(ob)
        done = prev_done
        #print('++==============================================+++doneq')
        #print(done)
        #nob = cob
        for _ in range(self.n_step):
            policy = (self.model.forward_policy(ob, done, 'pqvw'))
        # ob 4,10 done 1
            #print('+++++++++===============================+++++policyoutptexplore')
            #print(policy)
            action = []
            for pi in policy:
                #print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%piforloop')
                #print(pi)
                epsilon = np.random.uniform(0, 1)
                #print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%epsilon')
                #print(epsilon)

                if epsilon > 0.3:
                    action.append(np.argmax(abs(np.array(pi))))
                    #print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%nonexpaction')
                    #print(action)
                else:
                    with tf.Session() as sess:
                        action.append(np.argmax(sess.run(gumbel_softmax(pi, hard=True))))
                        #print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%expaction')
                        #print(action)
            #print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%finalaction')
            #print(action) # action shape is 1,4
            next_ob, reward, done, global_reward = self.env.step(action)
            # next_ob 4,10 ; ob 4,10 ; reward 4 ; global_reward []
            #print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%nextobstep')
            #print(np.array(next_ob).shape)
            #print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%obstep')
            #print(np.array(ob).shape)
            #print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%reward')
            #print(np.array(reward).shape)
            #print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%globalreward')
            #print(np.array(done).shape)
            global_step = self.global_counter.next()
            self.cur_step += 1
            self.model.add_transition(ob, action, next_ob, reward, done)

            if self.global_counter.should_log() and self.agent.endswith('ddpg'):
                logging.info('''Training: global step %d, episode step %d,
                                   ob: %s, a: %s, nob: %s, pi: %s, t_pi: %s, v: %s, t_v:%s, r: %.2f, train r: %.2f, done: %r''' %
                             (global_step, self.cur_step,
                              str(ob), str(action), str(next_ob), str(policy), str(t_policy), str(value), str(t_value), global_reward, np.mean(reward), done))

            # 4 trans_buffer_ls objects are created, with trans_ls_buffer[0] having
            #(1,10) obs and nobs, (1) reward, [1] done for the first agent
            ob = next_ob

    def perform(self, test_ind, demo=False):
        ob = self.env.reset(gui=demo, test_ind=test_ind)
        nob = self.env.reset(gui=demo, test_ind=test_ind)
        # note this done is pre-decision to reset LSTM states!
        done = True
        self.model.reset()
        rewards = []
        while True:
            if self.agent == 'greedy':
                action = self.model.forward(ob)
            elif self.agent.endswith('ddpg'):  # add for actor network one hot
                policy = self.model.forward(nob, ob, done, 'p')
                if self.agent == 'maddpg':  #TODO: Change to maddpg
                    self.env.update_fingerprint(policy)  # where the fuck is action for ma2c
                if self.agent == 'iddpg':
                    action = np.argmax(np.array(policy))
                else:
                    action = []
                    for pi in policy:
                        action.append(np.argmax(np.array(pi)))
            else:
                action, _ = self.model.forward(ob)
            next_ob, reward, done, global_reward = self.env.step(action)
            rewards.append(global_reward)
            if done:
                break
            ob = next_ob
        mean_reward = np.mean(np.array(rewards))
        std_reward = np.std(np.array(rewards))
        return mean_reward, std_reward

    def run_thread(self, coord):
        '''Multi-threading is disabled'''
        ob = self.env.reset()
        done = False
        cum_reward = 0
        while not coord.should_stop():
            ob, done, R, cum_reward =self.explore(ob, done, cum_reward)
            global_step = self.global_counter.cur_step
            if self.agent.endswith('a2c'):
                self.model.backward(R, self.summary_writer, global_step)
            else:
                self.model.backward(self.summary_writer, global_step)
            self.summary_writer.flush()
            if (self.global_counter.should_stop()) and (not coord.should_stop()):
                self.env.terminate()
                coord.request_stop()
                logging.info('Training: stop condition reached!')
                return

    def run(self):
        while not self.global_counter.should_stop():
            # test
            if self.run_test and self.global_counter.should_test():
                rewards = []
                global_step = self.global_counter.cur_step
                self.env.train_mode = False
                for test_ind in range(self.test_num):
                    mean_reward, std_reward = self.perform(test_ind)
                    self.env.terminate()
                    rewards.append(mean_reward)
                    log = {'agent': self.agent,
                           'step': global_step,
                           'test_id': test_ind,
                           'avg_reward': mean_reward,
                           'std_reward': std_reward}
                    self.data.append(log)
                avg_reward = np.mean(np.array(rewards))
                self._add_summary(avg_reward, global_step, is_train=False)
                logging.info('Testing: global step %d, avg R: %.2f' %
                             (global_step, avg_reward))
            # train
            self.env.train_mode = True

            ob = self.env.reset()

            # cob = self.env.reset()

            # cob = ob
            # note this done is pre-decision to reset LSTM states!
            done = True
            self.model.reset()
            self.cur_step = 0
            rewards = []
            self.explore(ob, done)
            while True:
                obs, action, nob, cur_rewards, done = self.model.sample_transition()
                a_prime = []
                for j in range(self.n_step):
                    a = []
                    d = []
                    for i in range(4):
                        a.append(nob[i][j])
                        d.append(done[i][j])
                    #print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%a')
                    #print(a)
                    #print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%a')
                    #print(d)

                    a_prime.append((self.model.forward_t_policy(a, d)))
                #print('%%%%%%%%%%%%%%%%%%%%%%%%%%%awwwwwwww')
                #print(a_prime)

                #print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%aprime')
                #print(a_prime)
                #print(np.argmax(np.array(a_prime[0][0])))
                a_t = []
                for i in range(self.n_step):
                    a_p = []

                    for j in range(4):
                        a_p.append(np.argmax(np.array(a_prime[i][j])))
                    # print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%target_action')
                    a_t.append(a_p)
                #print(a_t)
                #print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%obs')
                #print(nob)
                #print(done[0])
                q_t = []
                for j in range(self.n_step):
                    a = []
                    d = []
                    tara = a_t[j]
                    for i in range(4):
                        a.append(nob[i][j])
                        d.append(done[i][j])
                    q_t.append(self.model.forward_t_value(a, tara, d))
                #print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%targetq')
                q_t = tf.squeeze((q_t))
                #print((q_t).shape)

                #print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%reward')
                #print(np.array(cur_rewards).shape)
                #print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%done')
                #print(np.array(done).shape)

                # for i in range(10):
                # self.model.forward_t_value(a_t[i],)

                #print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%reward')
                #print(cur_rewards[0])
                #y = []
                #print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%q_t')
                q_t = tf.transpose((q_t))
                #print(np.array(q_t).shape)
                y = []

                for i in range(4):
                    y.append(cur_rewards[i] + 0.99 * (1 - done[i]) * q_t[i])
                #print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%target_valueineedd')
                sess = tf.Session()
                y =  sess.run(y)
                #print(y)
                rewards += cur_rewards
                global_step = self.global_counter.cur_step

                if self.agent.endswith('ddpg'):
                    #print('+++++++++++++++++++++++++++++++++summarywrter')
                    #print(self.summary_writer)
                    #print('+++++++++++++++++++++++++++++++++++++glblstep')
                    #print(global_step)
                    self.model.backward_policy(y, obs, action, nob, cur_rewards, done, self.summary_writer, global_step)
                    self.model.backward_value(y, obs, action, nob, cur_rewards, done, self.summary_writer, global_step)
                else:
                    self.model.backward(self.summary_writer, global_step)
                # termination
                if done:
                    self.env.terminate()
                    #print('+++++++++++++++++++++++++++done=true')
                    break
            #print('+++++++++++++++++++++++reached mean reward')
            rewards = np.array(rewards)
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            #print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%actorparamsaccess')

            log = {'agent': self.agent,
                   'step': global_step,
                   'test_id': -1,
                   'avg_reward': mean_reward,
                   'std_reward': std_reward}
            self.data.append(log)
            self._add_summary(mean_reward, global_step)
            self.summary_writer.flush()
        df = pd.DataFrame(self.data)
        df.to_csv(self.output_path + 'train_reward.csv')

        #self.env.train_mode = True

        '''ob = self.env.reset()
        #gives 4,10 obs for 1,10 each agent
        #cob = self.env.reset()

        #cob = ob
        # note this done is pre-decision to reset LSTM states!
        done = True
        # for 1 step 1 done
        self.model.reset()
        self.cur_step = 0
        rewards = []

        self.explore(ob, done)
        state, action, next_state, reward, done = self.model.sample_transition()
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%state')
        print(state)
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%action')
        print(action)
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%nextstate')
        print(next_state)
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%reward')
        print(reward)
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%done')
        print(done)

        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%nextstatei')
        print(action)
        a_prime =[]
        for j in range(10):
            a =[]
            d = []
            for i in range(4):
                a.append(next_state[i][j])
                d.append(done[i][j])
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%a')
            print(a)
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%a')
            print(d)

            a_prime.append((self.model.forward_t_policy(a, d)))
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%awwwwwwww')
        print(a_prime)

        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%aprime')
        print(a_prime)
        print(np.argmax(np.array(a_prime[0][0])))
        a_t = []
        for i in range(10):
            a_p = []

            for j in range(4):
                a_p.append(np.argmax(np.array(a_prime[i][j])))
            #print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%target_action')
            a_t.append(a_p)
        print(a_t)
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%obs')
        print(next_state)
        print(done[0])
        q_t =[]
        for j in range(10):
            a =[]
            d = []
            tara = a_t[j]
            for i in range(4):
                a.append(next_state[i][j])
                d.append(done[i][j])
            q_t.append(self.model.forward_t_value(a, tara, d))
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%targetq')
        q_t = tf.squeeze((q_t))
        print((q_t).shape)

        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%reward')
        print(np.array(reward).shape)
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%done')
        print(np.array(done).shape)


        #for i in range(10):
        #self.model.forward_t_value(a_t[i],)


        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%reward')
        print(reward[0])
        y=[]
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%q_t')
        q_t = tf.transpose((q_t))
        print(np.array(q_t).shape)
        y =[]

        for i in range(4):
            y.append(reward[i] + 0.99 * (1 - done[i]) * q_t[i])
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%target_value')
        print(y)

        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%state')
        print(state)
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%action')
        print(action)

        '''
        '''v = []
        for j in range(10):
            a =[]
            d = []
            act =[]
            for i in range(4):
                a.append(state[i][j])
                d.append(done[i][j])
                act.append(action[i][j])
                print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%shapecheck')
                print(a)
                print(d)
            v.append(self.model.forward_value(a, act, d))
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%targetvalue')
        print(v)
        '''

        #self.model.backward_policy(self, y, state, action, next_state, reward, done)


        '''for i in range(4):
            y = reward + 0.95*(1-done)*self.model.forward_t_value(next_state, self.model.forward_t_policy(next_state, done))
            vf_loss = tf.reduce_mean(tf.square(y - self.model.forward_value(obs, action)))
            pol_loss = -tf.reduce_mean(self.model.forward_critic(obs, done))
            w_v = tf.trainable_variables(scope = 'fplstm0a/v')
            grads_pol = tf.gradients(vf_loss, w_v)
            #grads_crt = tf.gradients(pol_loss,w)
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%grads_pol')
            print(grads_pol)'''









        '''v = np.squeeze(np.array(v))
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%v')
        print(np.array(y).shape)
        v = np.transpose(np.array(v))
        print(np.array(v).shape)
        vf_loss = []
        for i in range(4):
            vf_loss.append(tf.reduce_mean(tf.square(v[i] - tf.stop_gradient(y[i]))))
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%vf_loss')
        print(vf_loss[0])
        # tensor having value function loss for each agent
        grad_vf_loss = []
        sess = tf.Session()
        wts = tf.trainable_variables(scope = None)
        print(wts)

        for i in range(4):
            wts_v = tf.trainable_variables(scope='fplstm_' + str(i) + 'a/v')
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%wts')
            print(wts_v)
            vf_loss = tf.reduce_mean(tf.square(self.model.forward_value()))
            grads_vf_loss = sess.run(tf.gradients(vf_loss[i], wts_v, name='grad_vf'))
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%grads_vf_loss')
            print(grads_vf_loss)
            grads_vf_loss, self.grad_vf_loss_norm = tf.clip_by_global_norm(grad_vf_loss, 40)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
            #loss = self.optimizer.minimize(vf_loss[i])
            self._train_v = self.optimizer.apply_gradients(list(zip(grads_vf_loss, wts_v)))
            #self._train_v = self.optimizer.apply_gradients(grads_and_vars)
        #loss = sess.run(self._train_v)
        #q_actual = self.model.forward_value(state, )

        #a_prime.append(self.model.forward_t_policy(next_state[j][i], done[j][i]))
        #print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%a_prime')
        #print(a_prime)

        # state is array of 4 arrays each representing each agent's state
        # action is array of 4 arrays each representing each agent's action


        # one iteration gives minibatch size samples of one agent

        #a_prime.append(self.model.forward_t_policy(next_state[k], done[k]))


        for j in range(4):
            state, action, next_state, reward, done = self.model.sample_transition(j)
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%statesampl')
            print(np.array(state).shape)
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%actionsampl')
            print(np.array(action).shape)
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%next_statesampl')
            print(np.array(next_state).shape)
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%rewardsampl')
            print(np.array(reward).shape)
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%done')
            print(np.array(done).shape)
            a_prime.append(self.model.forward_t_policy(next_state, done, j))# oehotfromlogits
        print('+++++++++++++++++++++++++++++++++++++++++++alltargetacs')
        print(a_prime)
        state, action, next_state, reward, done = self.model.sample_transition(i)

        q_prime = self.model.forward_t_value(next_state, a_prime, done)
        print('+++++++++++++++++++++++++++++++++++++++++++targetq')
        print(q_prime)
        y = reward + 0.99 * (1 - done) * q_prime
        print('++++++++++++++++++++++++++++++++++++++++++++++++y')
        print(y)
        q = self.model.forward_value(state, action, done)
        print('++++++++++++++++++++++++++++++++++++++++++++++++q')
        print(q)
        vf_loss = tf.reduce_mean(tf.square(q - tf.stop_gradient(y)))
        wts_v = tf.trainable_variables(scope=None)
        grad_vf_loss = tf.gradients(vf_loss, wts_v, name='grad_vf')
        grad_vf_loss, self.grad_vf_loss_norm = tf.clip_by_global_norm(grad_vf_loss, 40)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        self._train_v = self.optimizer.apply_gradients(list(zip(grad_vf_loss, wts_v)))
        for j in range(4):
            a.append(self.model.forward_t_policy(next_state[j], done))  #
    # gumbel_softmax
        pol_loss = -tf.reduce_mean(self.model.forward_value(obs, a))
        wts_p = tf.trainable_variables(scope=None)
        grad_pol_loss = tf.gradients(pol_loss, wts_v, name='grad_vf')
        grad_pol_loss, self.grad_pol_loss_norm = tf.clip_by_global_norm(grad_pol_loss, 40)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        self._train_p = self.optimizer.apply_gradients(list(zip(grad_pol_loss, wts_p)))
        with tf.Session() as sess:
            sess.run(self._train_p)
            sess.run(self._train_v)
        #rewards = np.array(rewards)
        #mean_reward = np.mean(rewards)
        #std_reward = np.std(rewards)
        #log = {'agent': self.agent,
        #       'step': global_step,
        #       'test_id': -1,
        #       'avg_reward': mean_reward,
        #       'std_reward': std_reward}
        #self.data.append(log)
        #self._add_summary(mean_reward, global_step)
        #self.summary_writer.flush()
    #df = pd.DataFrame(self.data)
    #df.to_csv(self.output_path + 'train_reward.csv')'''



class Tester(Trainer):
    def __init__(self, env, model, global_counter, summary_writer, output_path):
        super().__init__(env, model, global_counter, summary_writer)
        self.env.train_mode = False
        self.test_num = self.env.test_num
        self.output_path = output_path
        self.data = []
        logging.info('Testing: total test num: %d' % self.test_num)

    def _init_summary(self):
        self.reward = tf.placeholder(tf.float32, [])
        self.summary = tf.summary.scalar('test_reward', self.reward)

    def run_offline(self):
        # enable traffic measurments for offline test
        is_record = True
        record_stats = False
        self.env.cur_episode = 0
        self.env.init_data(is_record, record_stats, self.output_path)
        rewards = []
        for test_ind in range(self.test_num):
            rewards.append(self.perform(test_ind))
            self.env.terminate()
            time.sleep(2)
            self.env.collect_tripinfo()
        avg_reward = np.mean(np.array(rewards))
        logging.info('Offline testing: avg R: %.2f' % avg_reward)
        self.env.output_data()

    def run_online(self, coord):
        self.env.cur_episode = 0
        while not coord.should_stop():
            time.sleep(30)
            if self.global_counter.should_test():
                rewards = []
                global_step = self.global_counter.cur_step
                for test_ind in range(self.test_num):
                    cur_reward = self.perform(test_ind)
                    self.env.terminate()
                    rewards.append(cur_reward)
                    log = {'agent': self.agent,
                           'step': global_step,
                           'test_id': test_ind,
                           'reward': cur_reward}
                    self.data.append(log)
                avg_reward = np.mean(np.array(rewards))
                self._add_summary(avg_reward, global_step)
                logging.info('Testing: global step %d, avg R: %.2f' %
                             (global_step, avg_reward))
                # self.global_counter.update_test(avg_reward)
        df = pd.DataFrame(self.data)
        df.to_csv(self.output_path + 'train_reward.csv')


class Evaluator(Tester):
    def __init__(self, env, model, output_path, demo=False):
        self.env = env
        self.model = model
        self.agent = self.env.agent
        self.env.train_mode = False
        self.test_num = self.env.test_num
        self.output_path = output_path
        self.demo = demo

    def run(self):
        is_record = True
        record_stats = False
        self.env.cur_episode = 0
        self.env.init_data(is_record, record_stats, self.output_path)
        time.sleep(1)
        for test_ind in range(self.test_num):
            reward, _ = self.perform(test_ind, demo=self.demo)
            self.env.terminate()
            logging.info('test %i, avg reward %.2f' % (test_ind, reward))
            time.sleep(2)
            self.env.collect_tripinfo()
        self.env.output_data()
