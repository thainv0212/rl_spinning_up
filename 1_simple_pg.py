import tensorflow as tf
import numpy as np
import gym
from gym.spaces import Discrete, Box


def mlp(x, sizes, activation=tf.nn.tanh, output_activation=None):
    for size in sizes[:-1]:
        print(x)
        x = tf.layers.dense(x, units=size, activation=activation)
    return tf.layers.dense(x, units=sizes[-1], activation=output_activation)


def train(env_name='CarPole-V0', hidden_sizes=[32], lr=1e-2,
          epochs=50, batch_size=5000, render=False):
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    # core of policy network
    obs_ph = tf.placeholder(shape=(None, obs_dim), dtype=tf.float32)
    logits = mlp(obs_ph, sizes=hidden_sizes + [n_acts])
    # make action selection op
    actions_0 = tf.multinomial(logits=logits, num_samples=1)
    actions = tf.squeeze(actions_0, axis=1)

    # make loss function
    weights_ph = tf.placeholder(shape=(None,), dtype=tf.float32)
    act_ph = tf.placeholder(shape=(None,), dtype=tf.int32)
    action_masks = tf.one_hot(act_ph, n_acts)
    log_probs = tf.reduce_sum(action_masks * tf.nn.log_softmax(logits), axis=1)
    loss = - tf.reduce_mean(weights_ph * log_probs)

    # make train op
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    def train_one_epoch():
        # make some empty lists for logging
        batch_obs = []
        batch_acts = []
        batch_weights = []
        batch_rets = []
        batch_lens = []

        # reset episode variables
        obs = env.reset()
        done = False
        ep_rews = []

        finished_rendering_this_epoch = False
        while True:
            if (not finished_rendering_this_epoch) and render:
                env.render()
            batch_obs.append(obs.copy())
            # act in the environment
            act_0, act = sess.run([actions_0, actions], {obs_ph: obs.reshape(1, -1)})
            # print('action', act_0, act)
            act = act[0]
            obs, rew, done, _ = env.step(act)

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)
            # save info about the episode if the episode finished
            if done:
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                batch_weights += [ep_ret] * ep_len

                obs, done, ep_rews = env.reset(), False, []

                finished_rendering_this_epoch = True

                if len(batch_obs) > batch_size:
                    break

        batch_loss, _ = sess.run([loss, train_op], {
            obs_ph: np.array(batch_obs),
            act_ph: np.array(batch_acts),
            weights_ph: np.array(batch_weights)
        })
        return batch_loss, batch_rets, batch_lens

    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f' % (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))


train('CartPole-v0', render=False, lr=0.01)
