#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 19:41:52 2018

@author: lihaoruo
"""

import threading
import numpy as np
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import gym
from atari_wrappers import wrap_deepmind
from time import sleep
from policy_fn import GreedyPolicy

GLOBAL_STEP = 0
def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
    op_holder = []
    
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

def process_frame(image):
    image = np.reshape(image,[np.prod(image.shape)]) / 255.0
    return image

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

class OC_Network():
    def __init__(self,s_size,a_size,scope,trainer):
        with tf.variable_scope(scope):
            self.inputs = tf.placeholder(shape=[None,s_size],dtype=tf.float32)
            self.imageIn = tf.reshape(self.inputs,shape=[-1,84,84,1])
            self.conv1 = slim.conv2d(activation_fn=tf.nn.relu,
                inputs=self.imageIn,num_outputs=32,
                kernel_size=[8,8],stride=[4,4],padding='VALID')
            self.conv2 = slim.conv2d(activation_fn=tf.nn.relu,
                inputs=self.conv1,num_outputs=64,
                kernel_size=[4,4],stride=[2,2],padding='VALID')
            self.conv3 = slim.conv2d(activation_fn=tf.nn.relu,
                inputs=self.conv2,num_outputs=64,
                kernel_size=[3,3],stride=[1,1],padding='VALID')
            hidden = slim.fully_connected(slim.flatten(self.conv3),512,activation_fn=tf.nn.relu)

            policy = slim.fully_connected(hidden ,a_size * num_options,
                activation_fn=None,
                weights_initializer=normalized_columns_initializer(0.01),
                biases_initializer=None)
            self.q_options = slim.fully_connected(hidden, num_options,
                activation_fn=None,
                weights_initializer=normalized_columns_initializer(1.0),
                biases_initializer=None)
            self.beta = slim.fully_connected(hidden, num_options,
                activation_fn=tf.nn.sigmoid,
                weights_initializer=normalized_columns_initializer(0.1),
                biases_initializer=None)
            self.policy = tf.nn.softmax(tf.reshape(policy, [-1, num_options, a_size]), dim=2)
            
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None,num_options,a_size],dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
                self.advantages_U = tf.placeholder(shape=[None],dtype=tf.float32)
                self.advantages_omg = tf.placeholder(shape=[None],dtype=tf.float32)
                self.betas_omg = tf.placeholder(shape=[None],dtype=tf.float32)
                self.options = tf.placeholder(shape=[None,num_options],dtype=tf.float32)
                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions, axis=2)
                self.responsible_output = tf.reduce_sum(self.responsible_outputs * self.options, axis=1)
                
                self.q_option = tf.reduce_sum(self.q_options * self.options, axis=1)
                self.value_loss = tf.reduce_sum(tf.square(self.target_v-self.q_option))
                self.entropy = tf.reduce_sum(self.policy * tf.log(self.policy))
                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_output)*self.advantages_U)
                self.beta_loss = tf.reduce_sum(self.advantages_omg * self.betas_omg)
                self.loss = 0.5 * self.value_loss + self.policy_loss + self.beta_loss + self.entropy * 0.01

                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss,local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,40.0)
                
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))

class Worker():
    def __init__(self,env,name,s_size,a_size,trainer,model_path,global_episodes):
        self.name = "worker_" + str(name)
        self.number = name        
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []

        self.local_OC = OC_Network(s_size,a_size,self.name,trainer)
        self.update_local_ops = update_target_graph('global',self.name)        
        self.env = env
        self.policy_fn = GreedyPolicy(epsilon=0.1, final_step=1e8, min_epsilon=0.01)
        
    def train(self,rollout,sess,gamma,returns):
        rollout   = np.array(rollout)
        betas_omg = rollout[:,0]
        q_options = rollout[:,1]
        actions   = rollout[:,2]
        rewards   = rollout[:,3]
        observations = rollout[:,4]
        is_initial_betas = rollout[:,5]
        options   = rollout[:,6]

        discounted_rewards = []
        for reward in rewards[::-1]:
            returns = reward + returns * gamma
            discounted_rewards.append(returns)
        discounted_rewards.reverse()
               
        q_options      = np.vstack(q_options)
        betas_omg      = betas_omg * (1-is_initial_betas)
        betas_OMG      = np.vstack(betas_omg)
        advantages_U   = discounted_rewards - q_options
        advantages_u   = np.zeros([batch_size])
        #advantages_OMG = q_options - np.max(q_options)
        advantages_OMG = np.zeros([batch_size, num_options])
        for i in range(batch_size):
            advantages_OMG[i] = q_options[i] - np.max(q_options[i])
        
        betas_omg         = np.zeros([batch_size])
        action            = np.zeros((batch_size, a_size))
        discounted_reward = np.zeros([batch_size])
        option            = np.zeros([batch_size, num_options])
        advantages_omg    = np.zeros([batch_size])
        action_now        = np.zeros((batch_size, num_options, a_size))
        
        for i in range(batch_size):
            betas_omg[i] = betas_OMG[i][options[i]]
            action[i][actions[i]] = 1
            discounted_reward[i] = discounted_rewards[i][options[i]]
            advantages_u[i] = advantages_U[i][options[i]]
            option[i][options[i]] = 1
            advantages_omg[i] = advantages_OMG[i][options[i]]
            for j in range(num_options):
                for k in range(a_size):
                    action_now[i][j][k] = action[i][k]

        feed_dict = {self.local_OC.actions:action_now,
                     self.local_OC.target_v:discounted_reward,
                     self.local_OC.inputs:np.vstack(observations),
                     self.local_OC.advantages_U:advantages_u,
                     self.local_OC.advantages_omg:advantages_omg,
                     self.local_OC.betas_omg:betas_omg,
                     self.local_OC.options:option}

        v_l,p_l,b_l,_ = sess.run([self.local_OC.value_loss,
                              self.local_OC.policy_loss,
                              self.local_OC.beta_loss,
                              self.local_OC.apply_grads], feed_dict=feed_dict)

        return v_l / len(rollout), p_l / len(rollout), b_l / len(rollout)
        
    def work(self,gamma,sess,coord,saver):
        global GLOBAL_STEP
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print ("Starting worker " + str(self.number))
        self.is_initial_betas = np.ones(1)
        with sess.as_default(), sess.graph.as_default():                 
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_reward = 0
                episode_step_count = 0
                terminals = False

                s = self.env.reset()
                s = process_frame(s)
                q_options,betas,pi=sess.run([self.local_OC.q_options,self.local_OC.beta,self.local_OC.policy],
                                            feed_dict={self.local_OC.inputs:[s]})
                q_options = q_options[0]
                betas = betas[0]
                pi = pi[0]
                options = self.policy_fn.sample(q_options)
                while not terminals:
                    GLOBAL_STEP += 1
                    pi = pi[options]
                    a = np.random.choice(pi,p=pi)
                    a = np.argmax(pi == a) 
                    s1, r, terminals, _ = self.env.step(a)
                    if terminals == False:
                        s1 = process_frame(s1)
                    else:
                        s1 = s
                        options = self.policy_fn.sample(q_options)
                        
                    episode_buffer.append([betas,q_options,a,r,s,self.is_initial_betas,options])
                    episode_reward += r
                    q_options_,betas_,pi_=sess.run([self.local_OC.q_options,self.local_OC.beta,self.local_OC.policy], 
                                                   feed_dict={self.local_OC.inputs:[s1]})
                    if betas[options] < 0.5: 
                        options = self.policy_fn.sample(q_options)

                    s = s1
                    q_options = q_options_[0]
                    betas = betas_[0]
                    pi = pi_[0]
                    self.is_initial_betas = np.asarray(terminals, dtype=np.float32)
                    total_steps += 1
                    episode_step_count += 1
                    self.policy_fn.update_epsilon()
                    
                    if len(episode_buffer) == batch_size and terminals != True:
                        q_options_next, betas_next = sess.run([self.local_OC.q_options, self.local_OC.beta], 
                                                              feed_dict={self.local_OC.inputs:[s]})
                        returns = (1 - betas_next) * q_options_next + betas_next * np.max(q_options_next,axis=1)
                        v_l,p_l,b_l = self.train(episode_buffer,sess,gamma,returns[0])
                        episode_buffer = []
                        sess.run(self.update_local_ops)
                    if terminals == True:
                        break
                
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                if episode_count % 5 == 0 and episode_count != 0:
                    if self.name == 'worker_0' and episode_count % 5 == 0:
                        print('\n episode: ', episode_count, 'global_step:', GLOBAL_STEP,\
                              'mean episode reward: ', np.mean(self.episode_rewards[-5:]))

                    print ('vloss:',v_l, 'ploss:',p_l, 'bloss:',b_l)
                    if episode_count % 100 == 0 and self.name == 'worker_0':
                        saver.save(sess,self.model_path+'/oc-'+str(episode_count)+'.cptk')
                        print ("Saved Model")
                episode_count += 1


def get_env(task):
    env_id = task.env_id
    env = gym.make(env_id)
    env = wrap_deepmind(env)
    return env

gamma = .99 
s_size = 7056
load_model = False
model_path = './ocmodel'
num_options = 4
batch_size = 10

benchmark = gym.benchmark_spec('Atari40M')
task = benchmark.tasks[3]

tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)
    
env = get_env(task)
a_size = env.action_space.n

global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
master_network = OC_Network(s_size,a_size,'global',None)
num_workers = 16
workers = []

for i in range(num_workers):
    env = get_env(task)
    workers.append(Worker(env,i,s_size,a_size,trainer,model_path,global_episodes))
saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    sess.run(tf.global_variables_initializer())
    if load_model == True:
        print ('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess,ckpt.model_checkpoint_path)

    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(gamma,sess,coord,saver)
        t = threading.Thread(target=(worker_work))
        t.start()
        sleep(0.5)
        worker_threads.append(t)
    coord.join(worker_threads)
    
