import numpy as np
from networks import action_to_one_hot

class Node(object):

    def __init__(self, prior):
        """
        Node in MCTS
        prior: The prior on the node, computed from policy network
        """
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_representation = None
        self.reward = 0
        self.expanded = False

    def value(self):
        """
        Compute value of a node
        """
        if self.visit_count == 0:
            return 0
        else:
            return self.value_sum / self.visit_count


def run_mcts(config, root, network, min_max_stats):
    """
    Main loop for MCTS for config.num_simulations simulations

    root: the root node
    network: the network
    min_max_stats: the min max stats object for the simulation

    Hint:
    The MCTS should capture selection, expansion and backpropagation
    """
    for i in range(config.num_simulations):
        history = []
        node = root
        search_path = [node]

        while node.expanded:
            action, node = select_child(config, node, min_max_stats)
            history.append(action)
            search_path.append(node)
        parent = search_path[-2]
        action = history[-1]
        value = expand_node(node, list(
            range(config.action_space_size)), network, parent.hidden_representation, action)
        backpropagate(search_path, value,
                      config.discount, min_max_stats)


def select_action(config, num_moves, node, network, test=False):
    """
    Select an action to take

    If in train mode: action selection should be performed stochastically
    with temperature t
    If in test mode: action selection should be performed with argmax
    """
    visit_counts = [
        (child.visit_count, action) for action, child in node.children.items()
    ]
    if not test:
        t = config.visit_softmax_temperature_fn(num_moves=num_moves)
        action = softmax_sample(visit_counts, t)
    else:
        action = softmax_sample(visit_counts, 0)
    return action


def select_child(config, node, min_max_stats):
    """
    TODO: Implement this function
    Select a child in the MCTS
    This should be done using the UCB score, which uses the
    normalized Q values from the min max stats
    """
    max_ucb = -float('inf')
    selected_action = None
    selected_child = None
    for action, child in node.children.items():
        score = ucb_score(config, node, child, min_max_stats)
        if score > max_ucb:
            max_ucb = score
            selected_action = action
            selected_child = child
    return selected_action, selected_child


    raise NotImplementedError()
    return action, child


def ucb_score(config, parent, child, min_max_stats):
    """
    Compute UCB Score of a child given the parent statistics
    """
    pb_c = np.log((parent.visit_count + config.pb_c_base + 1)
                  / config.pb_c_base) + config.pb_c_init
    pb_c *= np.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c*child.prior
    if child.visit_count > 0:
        value_score = min_max_stats.normalize(
            child.reward + config.discount*child.value())
    else:
        value_score = 0
    return prior_score + value_score

def softmax(x):
    # Stable softmax implementation
    x = np.array(x)
    x -= np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

def expand_root(node, actions, network, current_state):
    """
    TODO: Implement this function
    Expand the root node given the current state

    This should perform initial inference, and calculate a softmax policy over children
    You should set the attributes hidden representation, the reward, the policy and children of the node
    Also, set node.expanded to be true
    For setting the nodes children, you should use node.children and  instantiate
    with the prior from the policy

    Return: the value of the root
    """
    # Extract the actual observation array and reshape
    # print(current_state)
    current_state = np.expand_dims(current_state, axis=0)  # Add batch dimension
    hidden_representation = network.representation_network(current_state)
    value_logits = network.value_network(hidden_representation)
    policy_logits = network.policy_network(hidden_representation)

    # Convert policy_logits to a probability distribution
    policy = softmax(policy_logits[0].numpy())

    # Set root node attributes
    node.hidden_representation = hidden_representation
    node.reward = 0.0
    node.expanded = True
    node.policy = policy

    # Create children for each action
    for a, p in zip(actions, policy):
        node.children[a] = Node(p)

    # Return the value of the root
    # Since value network outputs logits corresponding to categorical representation of value
    # We select the highest probability index or use the full distribution if needed.
    # Depending on the setup, value_logits might need to be converted back to scalar value.
    # If you have a helper function for that, call it. Otherwise, just pick the expected value.
    # For simplicity, assume network._value_transform is available:
    return network._value_transform(value_logits)
    # get hidden state representation




def expand_node(node, actions, network, parent_state, parent_action):
    """
    TODO: Implement this function
    Expand a node given the parent state and action
    This should perform recurrent_inference, and store the appropriate values
    The function should look almost identical to expand_root

    Return: value
    """
    # Conditioned state: parent_state plus action one-hot
    action_one_hot = action_to_one_hot(parent_action, network.action_size)
    conditioned_state = np.concatenate([parent_state.numpy(), action_one_hot], axis=1)

    # Run dynamic network to get next hidden state and reward
    # conditioned_state shape: (batch, embedding + action_space)
    hidden_representation = network.dynamic_network(conditioned_state)
    reward = network.reward_network(hidden_representation)

    value_logits = network.value_network(hidden_representation)
    policy_logits = network.policy_network(hidden_representation)

    policy = softmax(policy_logits[0].numpy())

    node.hidden_representation = hidden_representation
    node.reward = reward.numpy().item()
    node.expanded = True
    node.policy = policy

    for a, p in zip(actions, policy):
        node.children[a] = Node(p)

    return network._value_transform(value_logits)

    raise NotImplementedError()
    return value


def backpropagate(path, value, discount, min_max_stats):
    """
    Backpropagate the value up the path

    This should update a nodes value_sum, and its visit count

    Update the value with discount and reward of node
    """
    for i in reversed(range(len(path))):
        node = path[i]
        node.visit_count += 1
        node.value_sum += value
        min_max_stats.update(node.value())

        if i > 0:
            # Incorporate the node's reward into the discounted value
            value = node.reward + discount * value

    # for node in reversed(path):
    #     # YOUR CODE HERE
    #     min_max_stats.update(node.value())

    # raise NotImplementedError()


def add_exploration_noise(config, node):
    """
    Add exploration noise by adding dirichlet noise to the prior over children
    This is governed by root_dirichlet_alpha and root_exploration_fraction
    """
    actions = list(node.children.keys())
    noise = np.random.dirichlet([config.root_dirichlet_alpha]*len(actions))
    frac = config.root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1-frac) + n*frac


def visit_softmax_temperature(num_moves):
    """
    This function regulates exploration vs exploitation when selecting actions
    during self-play.
    Given the current number of games played by the learning algorithm, return the
    temperature value to be used by MCTS.

    You are welcome to devise a more complicated temperature scheme
    """
    return 1


def softmax_sample(visit_counts, temperature):
    """
    Sample an actions

    Input: visit_counts as list of [(visit_count, action)] for each child
    If temperature == 0, choose argmax
    Else: Compute distribution over visit_counts and sample action as in writeup
    """
    if temperature == 0:
        # Pick the action with the highest visit count
        _, action = max(visit_counts, key=lambda x: x[0])
        return action
    else:
        counts = np.array([c for c, _ in visit_counts])
        actions = [a for _, a in visit_counts]
        counts_pow = counts ** (1.0 / temperature)
        probs = counts_pow / np.sum(counts_pow)
        return np.random.choice(actions, p=probs)

