import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MSE
from tensorflow.keras.regularizers import L2
import numpy as np
import math
from typing import Callable
from networks_base import BaseNetwork


def action_to_one_hot(action, action_space_size):
    """
    Compute one hot of action to be combined with state representation
    """
    return np.array([1 if i == action else 0 for i in range(action_space_size)]).reshape(1, -1)


class CartPoleNetwork(BaseNetwork):

    def __init__(self, action_size, state_shape, embedding_size, max_value):
        """
        Defines the CartPoleNetwork
        action_size: the number of actions
        state_shape: the shape of the input state
        embedding_size: the size of the embedding layer for representation
        max_value: denotes the max reward of a game for value transform
        """
        self.action_size = action_size
        self.state_shape = state_shape
        self.embedding_size = embedding_size
        self.train_steps = 0
        self.hidden_neurons = 48
        # value support size should be <= math.ceil(math.sqrt(max_value)) + 1
        self.value_support_size = math.ceil(math.sqrt(max_value)) + 1
        # Only in negative values are allowed
        self.full_support_size = 2 * self.value_support_size + 1
        regularizer = L2(1e-4)  # Weight Decay Strength

        # Create Networks, we use tanh activation as a normalizer on the state
        # Standard muzero performs max-min normalization
        representation_network = Sequential([Dense(self.hidden_neurons, activation='relu', kernel_regularizer=regularizer),
                                             Dense(self.embedding_size, activation='tanh',
                                                   kernel_regularizer=regularizer)])
        # Change to full support if negative values allowed
        value_network = Sequential([Dense(self.hidden_neurons, activation='relu', kernel_regularizer=regularizer),
                                    Dense(self.value_support_size, kernel_regularizer=regularizer)])

        policy_network = Sequential([Dense(self.hidden_neurons, activation='relu', kernel_regularizer=regularizer),
                                     Dense(action_size, kernel_regularizer=regularizer)])
        dynamic_network = Sequential([Dense(self.hidden_neurons, activation='relu', kernel_regularizer=regularizer),
                                      Dense(self.embedding_size, activation='tanh',
                                            kernel_regularizer=regularizer)])
        reward_network = Sequential([Dense(16, activation='relu', kernel_regularizer=regularizer),
                                     Dense(1, kernel_regularizer=regularizer)])

        super().__init__(representation_network, value_network,
                         policy_network, dynamic_network, reward_network)

    def training_steps(self):
        return self.train_steps

    def _value_transform(self, value_support) -> float:
        value = self._softmax(value_support)
        # Change to -value_support_size -> value_support_size for full support
        value = np.dot(value, range(self.value_support_size))
        value = tf.math.sign(value) * (
                ((tf.math.sqrt(1 + 4 * 0.001
                 * (tf.math.abs(value) + 1 + 0.001)) - 1) / (2 * 0.001)) ** 2 - 1
        )
        return value.numpy()[0]

    def _scalar_to_support(self, target_value):
        batch = len(target_value)
        targets = np.zeros((batch, self.value_support_size))
        target_value = tf.math.sign(target_value) * \
            (tf.math.sqrt(tf.math.abs(target_value) + 1)
             - 1 + 0.001*target_value)
        target_value = tf.clip_by_value(
            target_value, 0, self.value_support_size)
        floor = tf.math.floor(target_value)
        rest = target_value - floor
        targets[range(batch), tf.cast(floor, tf.int32)] = 1 - rest
        indexes = tf.cast(floor, tf.int32) + 1
        mask = indexes < self.value_support_size
        batch_mask = tf.boolean_mask(range(batch), mask)
        rest_mask = tf.boolean_mask(rest, mask)
        index_mask = tf.boolean_mask(indexes, mask)
        targets[batch_mask, index_mask] = rest_mask
        return targets

    def _reward_transform(self, reward) -> float:
        """
        No reward transform for cartpole
        """
        return np.asscalar(reward.numpy()[0])

    def _conditioned_hidden_state(self, hidden_state: np.array, action: int) -> np.array:
        """
        concatenate the hidden state and action for input to recurrent model
        """
        conditioned_hidden = tf.concat(
            (hidden_state, action_to_one_hot(action, self.action_size)), axis=1)
        return conditioned_hidden

    def _softmax(self, values):
        """
        Compute softmax
        """
        return tf.nn.softmax(values)

    def get_value_target(self, state):
        return self._value_transform(self.target_network.__call__(state))

    def cb_get_variables(self) -> Callable:
        """
        Return a callback that return the trainable variables of the network.
        """

        def get_variables():
            networks = (self.representation_network, self.value_network, self.policy_network,
                        self.dynamic_network, self.reward_network)
            return [variables
                    for variables_list in map(lambda n: n.trainable_weights, networks)
                    for variables in variables_list]

        return get_variables

    def save(self, path):
        """Save the networks."""
        if not os.path.isdir(path):
            os.mkdir(path)

        self.representation_network.save_weights(
            path + "/representation_net")  # , save_format='h5py')
        self.value_network.save_weights(
            path + "/value_net")  # , save_format='h5py')
        self.policy_network.save_weights(
            path + "/policy_net")  # , save_format='h5py')
        self.dynamic_network.save_weights(
            path + "/dynamic_net")  # , save_format='h5py')
        self.reward_network.save_weights(
            path + "/reward_net")  # , save_format='h5py')
        print("saved network at path:", path)

    def load(self, path):
        """Load previously stored network parameters."""
        self.built = True
        self.representation_network.built = True
        self.representation_network.load_weights(path + "/representation_net")
        self.value_network.load_weights(path + "/value_net")
        self.policy_network.load_weights(path + "/policy_net")
        self.dynamic_network.load_weights(path + "/dynamic_net")
        self.reward_network.load_weights(path + "/reward_net")
        print("loaded pre-trained weights at path:", path)


def scale_gradient(tensor, scale):
    """
    Function to scale gradient as described in MuZero Appendix
    """
    return tensor*scale + tf.stop_gradient(tensor) * (1. - scale)


def train_network(config, network, replay_buffer, optimizer, train_results):
    """
    Train Network for N steps
    """
    for _ in range(config.train_per_epoch):
        batch = replay_buffer.sample_batch()
        update_weights(config, network, optimizer, batch, train_results)

def update_weights(config, network, optimizer, batch, train_results):
    """
    TODO: Implement this function
    Train the network_model by sampling games from the replay_buffer.
    config: A dictionary specifying parameter configurations
    network: The network class to train
    optimizer: The optimizer used to update the network_model weights
    batch: The batch of experience
    train_results: The class to store the train results

    Hints:
    The network initial_model should be used to create the hidden state
    The recurrent_model should be used as the dynamics, which unroll in the latent space.

    You should accumulate loss in the value, the policy, and the reward (after the first state)
    Loss Note: The policy outputs are the logits, same with the value categorical representation
    You should use tf.nn.softmax_cross_entropy_with_logits to compute the loss in these cases
    """

    (state_batch, targets_init_batch, targets_recurrent_batch, actions_batch) = batch
    num_unroll_steps = config.num_unroll_steps

    # state_batch is a tuple/list of states, shape [batch_size, state_dim]
    state_batch = tf.convert_to_tensor(np.stack(state_batch, axis=0), dtype=tf.float32)
    batch_size = state_batch.shape[0]

    # targets_init_batch: [(value, _, policy), ...] for each sample in batch
    target_value_init, _, target_policy_init = zip(*targets_init_batch)
    target_value_init = tf.convert_to_tensor(target_value_init, dtype=tf.float32)  # shape (batch_size,)
    target_policy_init = tf.convert_to_tensor(target_policy_init, dtype=tf.float32) # shape (batch_size, action_size)
    # Convert scalar values to categorical support
    target_value_init = network._scalar_to_support(target_value_init)  # shape (batch_size, value_support_size)

    # targets_recurrent_batch is structured as:
    # targets_recurrent_batch: a tuple of length num_unroll_steps
    # Each element is a tuple of length batch_size, containing (value, reward, policy) for that step per sample.
    # actions_batch: a tuple of length num_unroll_steps
    # Each element is a tuple of length batch_size, containing the action for that step per sample.

    var_list = network.cb_get_variables()()

    with tf.GradientTape() as tape:
        # Initial inference
        hidden_representation = network.representation_network(state_batch)  # (batch_size, embedding)
        value_logits = network.value_network(hidden_representation)          # (batch_size, value_support_size)
        policy_logits = network.policy_network(hidden_representation)        # (batch_size, action_size)

        # Compute initial losses (no reward loss at initial step)
        value_loss_init = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=target_value_init, logits=value_logits)
        )
        policy_loss_init = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=target_policy_init, logits=policy_logits)
        )

        # Scale value loss by 0.25
        loss_total = 0.25 * value_loss_init + policy_loss_init

        total_value_loss = value_loss_init
        total_policy_loss = policy_loss_init
        total_reward_loss = 0.0

        # Unroll for num_unroll_steps
        # actions_batch and targets_recurrent_batch each have length num_unroll_steps.
        # For each unroll step, we have a tuple of batch_size elements.
        # We'll process each unroll step in batch form:
        for step in range(num_unroll_steps):
            # Extract actions for this step: shape (batch_size,)
            actions_step = actions_batch[step]
            actions_step = tf.convert_to_tensor(actions_step, dtype=tf.int32)

            # Extract targets (value, reward, policy) for this step
            # targets_recurrent_batch[step] is a tuple of length batch_size
            # Each element is (value, reward, policy)
            val_rew_pol_step = targets_recurrent_batch[step]

            # Unzip them into separate arrays
            target_values_step, target_rewards_step, target_policies_step = zip(*val_rew_pol_step)

            target_values_step = tf.convert_to_tensor(target_values_step, dtype=tf.float32)     # (batch_size,)
            target_rewards_step = tf.convert_to_tensor(target_rewards_step, dtype=tf.float32)   # (batch_size,)
            target_policies_step = tf.convert_to_tensor(target_policies_step, dtype=tf.float32) # (batch_size, action_size)

            # Convert value targets to support
            target_values_step = network._scalar_to_support(target_values_step) # (batch_size, value_support_size)

            # One-hot encode actions: (batch_size, action_size)
            act_one_hot = tf.one_hot(actions_step, network.action_size, dtype=tf.float32)

            # Concatenate hidden_representation with actions
            # hidden_representation: (batch_size, embedding)
            # act_one_hot: (batch_size, action_size)
            conditioned_input = tf.concat([hidden_representation, act_one_hot], axis=1) 

            # Dynamics + prediction (recurrent inference)
            next_hidden = network.dynamic_network(conditioned_input)    # (batch_size, embedding)
            # Scale gradient of hidden representation by 1/2
            next_hidden = scale_gradient(next_hidden, 0.5)

            reward = network.reward_network(next_hidden)                 # (batch_size, 1)
            value_logits = network.value_network(next_hidden)            # (batch_size, value_support_size)
            policy_logits = network.policy_network(next_hidden)          # (batch_size, action_size)

            # Compute losses at this step
            value_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=target_values_step, logits=value_logits)
            )
            policy_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=target_policies_step, logits=policy_logits)
            )
            # reward: (batch_size, 1), target_rewards_step: (batch_size,)
            # Expand target_rewards_step to (batch_size, 1) for broadcasting
            target_rewards_step = tf.reshape(target_rewards_step, [-1, 1])
            reward_loss = tf.reduce_mean(tf.square(reward - target_rewards_step))

            # Scale value loss by 0.25
            step_loss = 0.25 * value_loss + policy_loss + reward_loss
            # Scale gradient by 1/num_unroll_steps
            step_loss = step_loss / float(num_unroll_steps)

            # Accumulate losses
            loss_total += step_loss
            total_value_loss += value_loss / float(num_unroll_steps)
            total_policy_loss += policy_loss / float(num_unroll_steps)
            total_reward_loss += reward_loss / float(num_unroll_steps)

            # Update hidden_representation for next step
            hidden_representation = next_hidden

        # Record results
        train_results.total_losses.append(loss_total.numpy())
        train_results.value_losses.append(total_value_loss.numpy())
        train_results.policy_losses.append(total_policy_loss.numpy())
        train_results.reward_losses.append(total_reward_loss.numpy())

    
        grads = tape.gradient(loss_total, var_list)
        optimizer.apply_gradients(zip(grads, var_list))
        network.train_steps += 1




def update_weights__(config, network, optimizer, batch, train_results):
    """
    TODO: Implement this function
    Train the network_model by sampling games from the replay_buffer.
    config: A dictionary specifying parameter configurations
    network: The network class to train
    optimizer: The optimizer used to update the network_model weights
    batch: The batch of experience
    train_results: The class to store the train results

    Hints:
    The network initial_model should be used to create the hidden state
    The recurrent_model should be used as the dynamics, which unroll in the latent space.

    You should accumulate loss in the value, the policy, and the reward (after the first state)
    Loss Note: The policy outputs are the logits, same with the value categorical representation
    You should use tf.nn.softmax_cross_entropy_with_logits to compute the loss in these cases
    """
    # for every game in sample batch, unroll and update network_model weights
    def loss():
        loss = 0
        total_value_loss = 0
        total_reward_loss = 0
        total_policy_loss = 0
        (state_batch, targets_init_batch, targets_recurrent_batch,
         actions_batch) = batch

        # YOUR CODE HERE: Perform initial embedding of state batch

        target_value_batch, _, target_policy_batch = zip(
            *targets_init_batch)
        # Use this to convert scalar value targets to categorical representation
        # This now matches output of initial_model, and can be used with cross entropy loss
        target_value_batch = network._scalar_to_support(
            tf.convert_to_tensor(target_value_batch))
        # YOUR CODE HERE: Compute the loss of the first pass (no reward loss)
        # Remember to scale value loss!

        for actions_batch, targets_batch in zip(actions_batch, targets_recurrent_batch):
            target_value_batch, target_reward_batch, target_policy_batch = zip(
                *targets_batch)
            # YOUR CODE HERE:
            # Create conditioned_representation: concatenate representations with actions batch
            # Recurrent step from conditioned representation: recurrent + prediction networks

            # Same as above, convert scalar targets to categorical
            target_value_batch = tf.convert_to_tensor(target_value_batch)
            target_value_batch = network._scalar_to_support(target_value_batch)

            target_policy_batch = tf.convert_to_tensor(target_policy_batch)
            target_reward_batch = tf.convert_to_tensor(target_reward_batch)

            # YOUR CODE HERE: Compute value loss, reward loss, policy loss
            # Remember to scale value loss!
            # Add to total losses

            # YOUR CODE HERE: Half the gradient of the representation

            # YOUR CODE HERE: Sum the losses, scale gradient of the loss, add to overall loss
        train_results.total_losses.append(loss)
        train_results.value_losses.append(total_value_loss)
        train_results.policy_losses.append(total_policy_loss)
        train_results.reward_losses.append(total_reward_loss)
        return loss
    optimizer.minimize(loss=loss, var_list=network.cb_get_variables())
    network.train_steps += 1
    raise NotImplementedError()
