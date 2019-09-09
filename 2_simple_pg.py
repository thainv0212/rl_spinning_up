import tensorflow as tf
import numpy as np
import gym
from gym.spaces import Discrete, Box


def mlp(x, sizes, activation=tf.nn.tanh, output_activation=None):
    for size in sizes[:-1]:
        print(x)
        x = tf.layers.dense(x, units=size, activation=activation)
    return tf.layers.dense(x, units=sizes[-1], activation=output_activation)


def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i + 1] if i + 1 < n else 0)
    return rtgs


def train(env_name='CartPole-v0', hidden_sizes=[32], lr=1e-2,
          epochs=500, batch_size=5000, render=False):
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n
    # make core of policy network
    obs_ph = tf.placeholder(shape=(None, obs_dim), dtype=tf.float32)
    logits = mlp(obs_ph, sizes=hidden_sizes + [n_acts])
    actions_0 = tf.multinomial(logits=logits, num_samples=1)
    actions = tf.squeeze(actions_0, axis=1)

    weights_ph = tf.placeholder(shape=(None,), dtype=tf.float32)
    act_ph = tf.placeholder(shape=(None,), dtype=tf.int32)
    action_masks = tf.one_hot(act_ph, n_acts)
    log_probs = tf.reduce_sum(action_masks * tf.nn.log_softmax(logits), axis=1)
    loss = - tf.reduce_mean(weights_ph * log_probs)

    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    def train_one_epoch():
        batch_obs = []
        batch_acts = []
        batch_weights = []
        batch_rets = []
        batch_lens = []

        obs = env.reset()
        done = False
        ep_rews = []

        finished_rendering_this_episode = False
        while True:
            if (not finished_rendering_this_episode) and render:
                env.render()
            batch_obs.append(obs.copy())
            act = sess.run(actions, {obs_ph: obs.reshape(1, -1)})[0]
            obs, rew, done, _ = env.step(act)

            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                batch_weights += list(reward_to_go(ep_rews))
                obs, done, ep_rews = env.reset(), False, []
                finished_rendering_this_episode = True

                if len(batch_obs) > batch_size:
                    break

        batch_loss, _ = sess.run([loss, train_op], feed_dict={
            obs_ph: np.array(batch_obs),
            act_ph: np.array(batch_acts),
            weights_ph: np.array(batch_weights)
        })
        return batch_loss, batch_rets, batch_lens

    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        if i > 180:
            render = True
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f' % (
            i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))


train('CartPole-v1', render=False, lr=0.01)
