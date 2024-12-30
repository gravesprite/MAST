import numpy as np
from collections import OrderedDict
import copy
import torch
import time


class DecisionTreeNode:
    def __init__(self, args, left, right, depth):
        self.args = args

        # The information of the node, the left range, the right range, and the depth
        self.left, self.right = left, right
        self.depth = depth

        # store the reward that choose current node
        self.reward = 0

        # Store the sample time of the node
        self.sample_time = 0

        self.children = []

        self.can_choose = True
        if self.right - self.left == 1:
            self.can_choose = False

        # only useful for the node in the max depth
        self.choosen = []

        self.current_sample_id = -1


    def sample(self):

        sample_id = -1

        # check the sample availability
        if self.sample_time == self.right - self.left - 1:
            assert "Trying to sample the node with no available frames"

        if len(self.children) == 0:
            if self.depth == self.args.decision_tree_max_depth:
                choices = []
                for i in range(self.left + 1, self.right):
                    if i not in self.choosen:
                        choices.append(i)

                sample_id = np.random.choice(choices, 1)[0]

                self.choosen.append(sample_id)

            else:
                # Take the middle of the range
                sample_id = int((self.left +self.right) / 2)

                # Generate new node
                self.children.append(DecisionTreeNode(self.args, self.left, sample_id, self.depth + 1))
                self.children.append(DecisionTreeNode(self.args, sample_id, self.right, self.depth + 1))
        else:
            mab_values = []
            available_node_ind = []

            # Filter out nodes that can not be selected
            for i, node in enumerate(self.children):
                if node.can_choose:
                    available_node_ind.append(i)
            for node_ind in available_node_ind:
                node = self.children[node_ind]
                # print("reward: ", node.reward)
                # print("the weight for explo balance: ", np.sqrt(2 * np.log(self.sample_time + 1) / (node.sample_time + 1)))
                value = (node.reward + self.args.c_para *
                         np.sqrt(2 * np.log(self.sample_time + 1) / (node.sample_time + 1)))
                mab_values.append(value)

            selected_available_node_ind = np.argmax(mab_values)
            selected_node = available_node_ind[selected_available_node_ind]
            # print("left right ",  self.children[selected_node].left,  self.children[selected_node].right)
            sample_id = self.children[selected_node].sample()

        # Update the sample time
        self.sample_time += 1
        if self.sample_time == self.right - self.left - 1:
            self.can_choose = False

        self.current_sample_id = sample_id

        return sample_id

    def check_child(self, id):
        for i, node in enumerate(self.children):
            if node.left <= id and node.right >= id:
                return i

    def update_reward(self, reward):

        # Start testing different reward update methods
        self.reward = self.reward * 0.8 + reward * 0.2

        # The best reward update of ONCE dataset
        # self.reward = self.reward * 0.95 + reward * 0.05

        if len(self.children) == 0:
            return

        # Find the node id
        node_id = self.check_child(self.current_sample_id)

        # Update the child reward accordingly
        self.children[node_id].update_reward(reward)

        pass


class DecisionTree:
    def __init__(self, args, index, c_para=2):
        self.args = args

        self.index = index
        self.sample_list = index.sample_list
        self.sample_result = index.sample_result
        self.velocity_result = index.velocity_result

        self.c_para = c_para
        # This stores the decision made by the most recent sample function
        self.current_ind_choice = -1
        self.current_node = -1

        # Generate nodes w.r.t. the initial sample
        self.nodes = []

        for i in range(len(self.sample_list) - 1):
            start, end = self.sample_list[i], self.sample_list[i + 1]
            new_node = DecisionTreeNode(self.args, start, end, 0)
            self.nodes.append(new_node)

        self.total_sample_time = 0


        pass

    def sample(self):

        mab_values = []
        available_node_ind = []
        # Filter out nodes that can not be selected
        for i, node in enumerate(self.nodes):
            if node.can_choose:
                available_node_ind.append(i)
        for node_ind in available_node_ind:
            node = self.nodes[node_ind]
            # print("reward: ", node.reward)
            # print("the weight for explo balance: ", np.sqrt(2 * np.log(self.total_sample_time + 1) / (node.sample_time + 1)))
            # time.sleep(0.1)
            value = (node.reward + self.args.c_para *
                     np.sqrt(2 * np.log(self.total_sample_time + 1) / (node.sample_time + 1)))
            mab_values.append(value)

        selected_available_node_ind = np.argmax(mab_values)
        selected_node = available_node_ind[selected_available_node_ind]

        # for node in self.nodes:
        #     value = (node.reward + self.c_para *
        #              np.sqrt(2 * np.log(self.total_sample_time) / (node.sample_time + 0.1)))
        #     mab_values.append(value)
        #
        # selected_node = np.argmax(mab_values)

        sample_ind = self.nodes[selected_node].sample()

        self.current_ind_choice = sample_ind
        self.current_node = selected_node

        # Add one more sample count
        self.total_sample_time += 1

        return sample_ind

    def compute_reward(self, pred_result, proxy_result):

        reward = 0

        velocity = self.index.velocity_compute(pred_result, proxy_result, 1)

        velocity_frame_1 = velocity["velocity_of_box_frame_1"]
        additional_boxes = velocity["additional_boxes"]
        add_frame1 = velocity["additional_frame_1_id"]

        # print(velocity_frame_1)
        # time.sleep(0.1)

        # Return reward = 0, if the velocity_frame_1 does not contain frame
        if len(velocity_frame_1) == 0:
            return reward

        # print(velocity_frame_1)

        for v in velocity_frame_1:
            distance = torch.norm(v).item()
            reward += distance
        reward = reward / len(velocity_frame_1)
        reward /= self.args.detect_radius

        number_reward = 0
        if len(additional_boxes) != 0 or len(add_frame1) != 0:
            # number_reward = (len(additional_boxes) + len(add_frame1)) / (len(velocity_frame_1) + len(additional_boxes) + len(add_frame1) )
            number_reward = (len(additional_boxes) + len(add_frame1))

        # print(reward, number_reward)
        reward = (1 - self.args.reward_number_factor) * reward + self.args.reward_number_factor * number_reward

        return reward

    def generate_proxy_result(self, choose_ind):

        proxy_result = self.index.generate_proxy_result(choose_ind)

        return proxy_result

    def update(self, pred_result):
        pred_result = copy.deepcopy(pred_result)

        choose_ind = self.current_ind_choice
        choose_node_ind = self.current_node

        proxy_result = self.generate_proxy_result(choose_ind)

        reward = self.compute_reward(pred_result, proxy_result)

        node = self.nodes[choose_node_ind]

        node.update_reward(reward)
        pass
