"""
This file is for accurately sample the PC frames and manage the sample indexes, we first implement some basic
methods, such as the uniform sample, with specific budget.

"""
import numpy as np
from collections import OrderedDict
import torch
from pcdet.models import load_data_to_gpu

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import numpy as np
import math
from decision_tree import DecisionTree
import copy

def move_to_cpu(dict_obj):
    for key, value in dict_obj.items():
        if isinstance(value, torch.Tensor):
            dict_obj[key] = value.cpu()
    return dict_obj


class SampleIndex():

    def __init__(self, args, pred_model, dataloader, query, constant_param = 2, index_mode="no_velocity", pred_confidence=0.5, detect_radius=75):
        self.args = args

        self.constant_param = constant_param

        self.pred_model = pred_model

        # The model only consider the predicted boxes larger than the constant
        self.pred_confidence = pred_confidence

        # The model only consider the boxes within a specific spatial range, detect_radius
        self.detect_radius = detect_radius

        self.dataloader = dataloader
        self.query = query

        # Indicating the indexing model ["no_velocity", "velocity]
        self.index_mode = index_mode


        # Store the sample frame list and the result of model inference
        self.sample_list = []
        self.sample_result = {}

        # utilize the velocity information to help approximate query production
        self.velocity_result = {}

        self.distance_situation = {}

    def generate_meta_data(self):

        box_count = 0
        mean_distance = 0
        max_distance = 0

        for ind in self.sample_list:
            result = self.sample_result[ind]
            boxes = result['pred_boxes']

            for box in boxes:

                box_count += 1

                distance = math.sqrt(box[0] * box[0] + box[1] * box[1] + box[2] * box[2])

                if max_distance < distance:
                    max_distance = distance
                mean_distance = (1 / box_count) * distance + ((box_count - 1) / box_count) * mean_distance

        print("box_count: ", box_count)
        print("max_distance: ", max_distance)
        print("mean_distance: ", mean_distance)
        return

    def generate_proxy_result(self, sample_ind):

        self.sample_sort()
        # print(sample_ind)
        list_ind = self.sample_list.index(sample_ind)

        # get the frame before
        sample_ind_before = list_ind - 1
        sample_ind_before = self.sample_list[sample_ind_before]

        # print("sample_ind_before: ", sample_ind_before)
        # print("velocity: ", self.velocity_result[sample_ind_before])

        pred_result_before = copy.deepcopy(self.sample_result[sample_ind_before])
        velocity_before = copy.deepcopy(self.velocity_result[sample_ind_before])

        # print(velocity_before)

        for box in velocity_before["additional_boxes"]:
            pred_result_before['pred_boxes'].append(box[0])
            pred_result_before['pred_labels'].append(box[2])
            # pred_result_before['pred_scores'].append(torch.tensor(self.pred_confidence))
            pred_result_before['pred_scores'].append(0)

        steps = sample_ind - sample_ind_before

        score_change = 1 / steps

        for i in range(steps):
            pred_result_before = self.query.update_result_with_velocity(pred_result_before, velocity_before, score_change)

        proxy_result = pred_result_before
        # print(pred_result_before)
        return proxy_result

    def update_distance_situation(self, distance_situation, velocity, score_change):

        factor = 0.35

        new_distance_situation = copy.deepcopy(distance_situation)

        new_distance_situation["distance"] = [ new_distance_situation["distance"][i] + new_distance_situation['distance_change'][i]
                                           for i in range(len(new_distance_situation["distance"]))]

        for box_id in velocity['additional_frame_1_id']:
            new_distance_situation['score'][box_id] = max(torch.tensor(0), new_distance_situation['score'][box_id] - score_change * factor)
        for box_id in range(len(velocity["velocity_of_box_frame_1"]), len(velocity["velocity_of_box_frame_1"]) + len(velocity["additional_boxes"])):
            new_distance_situation['score'][box_id] = min(torch.tensor(1), new_distance_situation['score'][box_id] + score_change / factor)

        return new_distance_situation

    def predict_all_distance(self):

        new_distance_situations = {}

        for i in range(len(self.sample_list) - 1):

            sample_ind = self.sample_list[i]
            sample_ind_next = self.sample_list[i + 1]

            velocity = self.velocity_result[self.sample_list[i]]
            pred_result = self.sample_result[self.sample_list[i]]

            # velocity = copy.deepcopy(index.velocity_result[index.sample_list[i]])
            # pred_result = copy.deepcopy(index.sample_result[index.sample_list[i]])

            distance_situation = self.distance_situation[self.sample_list[i]]

            # change the confidence score
            score_change = 1 / (sample_ind_next - sample_ind)
            total_step = 0
            # print(distance_situation)
            for step in range(sample_ind_next - sample_ind - 1):
                distance_situation = self.update_distance_situation(distance_situation, velocity, score_change)
                # print(sample_ind_next, step, distance_situation)
                total_step += 1

                new_distance_situations[sample_ind + total_step] = distance_situation

        self.distance_situation = {**self.distance_situation, **new_distance_situations}
        return

    def generate_distance(self):

        if self.index_mode != 'velocity':
            assert "generate distance when the index mode is not velocity"

        for i in range(len(self.sample_list) - 1):
            self.distance_situation[self.sample_list[i]] = {}

        for id in self.sample_list[0: -1]:
            self.distance_situation[id]["distance"] = []

            self.distance_situation[id]['distance_change'] = []
            self.distance_situation[id]['score'] = []

            velocity_frame = self.velocity_result[id]

            for i, box in enumerate(self.sample_result[id]['pred_boxes']):
                x, y = box[0].item(), box[1].item()
                # distance = math.sqrt(x ** 2 + y ** 2)
                distance = math.hypot(x, y)

                self.distance_situation[id]["distance"].append(distance)

                velocity = velocity_frame["velocity_of_box_frame_1"][i]
                velocity = velocity[0:2]

                # Compute the direction vector of the line (from root_point to point)
                line_vector = torch.tensor([box[0].item(), box[1].item()])

                # Normalize the direction vector to get the unit vector
                line_unit_vector = line_vector / torch.norm(line_vector)

                # Compute the dot product of the velocity vector and the direction unit vector
                dot_product = torch.dot(velocity, line_unit_vector)

                scalar_projection = dot_product.item()

                self.distance_situation[id]['distance_change'].append(scalar_projection)

                self.distance_situation[id]['score'].append(self.sample_result[id]['pred_scores'][i])

            additional_boxes = velocity_frame["additional_boxes"]
            for box in additional_boxes:
                x, y = box[0][0].item(), box[0][1].item()
                distance = math.hypot(x, y)
                self.distance_situation[id]["distance"].append(distance)
                self.distance_situation[id]['distance_change'].append(0)
                self.distance_situation[id]['score'].append(0)
        pass



    """This function takes two frammes predict result as the input and the time gap
    between two frames
    """
    def velocity_compute(self, frame_result_1, frame_result_2, time_gap):

        frame_result_1 = copy.deepcopy(frame_result_1)
        frame_result_2 = copy.deepcopy(frame_result_2)

        # Extract bounding boxes and scores from the frame results
        boxes_1_list, scores_1, labels_1 = frame_result_1['pred_boxes'], frame_result_1['pred_scores'], frame_result_1[
            'pred_labels']
        boxes_2_list, scores_2, labels_2 = frame_result_2['pred_boxes'], frame_result_2['pred_scores'], frame_result_2[
            'pred_labels']

        # Initialize velocity dictionary
        velocities = {"velocity_of_box_frame_1": [None] * len(boxes_1_list), "additional_frame_1_id":[],"additional_boxes": []}

        # Get unique labels
        unique_labels = np.unique(np.concatenate([labels_1, labels_2]))

        if len(boxes_1_list) > 0:
            boxes_1 = torch.stack(boxes_1_list)
        else:
            if len(boxes_2_list) > len(boxes_1_list):
                boxes_2_label = torch.stack(boxes_2_list)
                centers_2 = boxes_2_label[:, :3]
            # Boxes in frame_2 that were not matched
                for j in set(range(len(boxes_2_list))):
                    # Calculate the velocity such that the box came from self.detect_radius meters away from the current position
                    direction = centers_2[j] / np.linalg.norm(centers_2[j])
                    original_position = centers_2[j] + direction * self.detect_radius

                    # Get the original box
                    original_box = boxes_2_label[j].clone()
                    # Update the center of the box
                    original_box[:3] = original_position

                    # velocities["additional_boxes"].append((original_box, - direction * self.detect_radius *
                    #                                        self.args.move_distance_ratio / time_gap, labels_2[j]))
                    velocities["additional_boxes"].append((original_box, 0, labels_2[j]))

            return velocities

        if len(boxes_2_list) > 0:
            try:
                boxes_2 = torch.stack(boxes_2_list)
            except:
                print("boxes_2_list is troublesome\n", type(boxes_2_list))
        else:
            if len(boxes_1_list) > len(boxes_2_list):
                boxes_1_label = torch.stack(boxes_1_list)
                centers_1 = boxes_1_label[:, :3]

                # Boxes in frame_1 that were not matched
                for i in set(range(len(boxes_1))):
                    # Calculate the velocity such that the box ends up 50 meters away from the center
                    direction = centers_1[i] / np.linalg.norm(centers_1[i])

                    # velocities["velocity_of_box_frame_1"][i] = (direction * self.detect_radius *
                    #                                             self.args.move_distance_ratio / time_gap)
                    velocities["velocity_of_box_frame_1"][i] = torch.tensor([0.0,0.0,0.0])
                    velocities["additional_frame_1_id"].append(i)

            return velocities

        # Iterate over each unique label
        for label in unique_labels:
            # Filter boxes with current label and keep track of original indices
            indices_1 = np.where(labels_1 == label)[0]
            indices_2 = np.where(labels_2 == label)[0]

            indices_1 = torch.tensor(indices_1)
            boxes_1_label = boxes_1[indices_1]

            indices_2 = torch.tensor(indices_2)
            boxes_2_label = boxes_2[indices_2]

            # Calculate the centers of the boxes in both frames
            centers_1 = boxes_1_label[:, :3]
            centers_2 = boxes_2_label[:, :3]

            # Calculate the pairwise distances between centers of boxes in both frames
            cost_matrix = cdist(centers_1, centers_2)

            # Use the Hungarian algorithm to find the optimal assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Iterate over each pair of matched boxes
            for i, j in zip(row_ind, col_ind):
                # Calculate the displacement between the centers of the boxes
                displacement = centers_2[j] - centers_1[i]

                # Calculate the velocity
                velocity = displacement / time_gap

                # Store the velocity in the original index
                velocities["velocity_of_box_frame_1"][indices_1[i]] = velocity

            # Handle unmatched boxes
            if len(boxes_1_label) > len(boxes_2_label):
                # Boxes in frame_1 that were not matched
                for i in set(range(len(boxes_1_label))) - set(row_ind):
                    # # Calculate the velocity such that the box ends up 50 meters away from the center
                    # direction = centers_1[i] / np.linalg.norm(centers_1[i])
                    # velocities["velocity_of_box_frame_1"][indices_1[i]] = direction * self.detect_radius * self.args.move_distance_ratio / time_gap

                    velocities["velocity_of_box_frame_1"][indices_1[i]] = torch.tensor([0.0,0.0,0.0])
                    velocities["additional_frame_1_id"].append(indices_1[i])
            elif len(boxes_2_label) > len(boxes_1_label):
                # Boxes in frame_2 that were not matched
                for j in set(range(len(boxes_2_label))) - set(col_ind):
                    # Calculate the velocity such that the box came from 50 meters away from the current position
                    direction = centers_2[j] / np.linalg.norm(centers_2[j])
                    original_position = centers_2[j] + direction * self.detect_radius

                    # Get the original box
                    original_box = boxes_2_label[j].clone()
                    # Update the center of the box
                    original_box[:3] = original_position

                    # velocities["additional_boxes"].append((original_box, -direction * self.detect_radius * self.args.move_distance_ratio / time_gap, label))
                    velocities["additional_boxes"].append((original_box,
                                                           0,
                                                           label))

        return velocities

    def init_label_distances(self, frame_indices):
        cluster_dict = OrderedDict()

        for i in range(len(frame_indices) - 1):
            start, end = frame_indices[i], frame_indices[i+1]
            corresponding_reps = []
            corresponding_reps.append(start)
            corresponding_reps.append(end)

            cache = []

            for rep in corresponding_reps:
                cache.append(self.query.compute_num(self.sample_result[rep]))

            cluster_dict[(start, end)] = {
                'members': corresponding_reps,
                'cache': cache,
                'distance': np.var(cache)
            }

        return cluster_dict

    def uniform_sample(self, budget):
        frame_num = int(len(self.dataloader.get_file_list()) * self.args.process_percentage)
        self.sample_list += np.linspace(0, frame_num -1, budget).astype(
            int).tolist()  # Uniformly sample the middle frames
        # print(self.sample_list)
        return

    def update_velocity(self, rep_idx):
        self.sample_sort()
        sample_ind = rep_idx
        sample_ind_in_list = self.sample_list.index(sample_ind)
        sample_ind_before = self.sample_list[sample_ind_in_list - 1]
        sample_ind_after = self.sample_list[sample_ind_in_list + 1]

        velocity_sample_before = self.velocity_compute(self.sample_result[sample_ind_before],
                                                       self.sample_result[sample_ind],
                                                       sample_ind - sample_ind_before)
        velocity_sample = self.velocity_compute(self.sample_result[sample_ind],
                                                self.sample_result[sample_ind_after],
                                                sample_ind_after - sample_ind)

        self.velocity_result[sample_ind_before] = velocity_sample_before
        self.velocity_result[sample_ind] = velocity_sample
        return

    def update(self, cluster_dict, cluster_key, rep_idx):
        start_idx, end_idx = cluster_key
        cluster_dict[(start_idx, end_idx)]['members'].append(rep_idx)

        data_dict = self.dataloader.get_item_ind(rep_idx)
        data_dict = self.dataloader.collate_batch([data_dict])
        load_data_to_gpu(data_dict)
        pred_dicts, _ = self.pred_model.forward(data_dict)
        pred_dicts = move_to_cpu(pred_dicts[0])

        # Filter out boxes with low confidence
        pred_dicts = self.filter_out_object_low_confidence(pred_dicts)

        # Store the result in the sample_result dictionary
        self.sample_result[rep_idx] = pred_dicts

        # update the velocity, update the frame before the sampled frame, and the sampled frame itself
        if self.index_mode == "velocity":
            self.update_velocity(rep_idx)


        cluster_dict[(start_idx, end_idx)]['cache'].append((self.query.compute_num(self.sample_result[rep_idx])))
        cluster_dict[(start_idx, end_idx)]['distance'] = np.var(cluster_dict[(start_idx, end_idx)]['cache'])

        return cluster_dict

    def mab_sample(self, budget):
        # get the number of total frames
        frame_num = len(self.dataloader.get_file_list())

        # use 1/4 budget to start the sample
        start_budget = budget // 4
        self.uniform_sample(start_budget)

        # Inference the result of all sampled frames
        for sample_ind in self.sample_list:
            data_dict = self.dataloader.get_item_ind(sample_ind)
            data_dict = self.dataloader.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = self.pred_model.forward(data_dict)
            pred_dicts = move_to_cpu(pred_dicts[0])

            # Filter out boxes with low confidence
            pred_dicts = self.filter_out_object_low_confidence(pred_dicts)

            # Store the result in the sample_result dictionary
            self.sample_result[sample_ind] = pred_dicts

        # initialize the velocity
        if self.index_mode == "velocity":
            for i, sample_ind in enumerate(self.sample_list):
                if i == len(self.sample_list) - 1:
                    break
                sample_ind_2 = self.sample_list[i + 1]
                frame_result_1 = self.sample_result[sample_ind]
                frame_result_2 = self.sample_result[sample_ind_2]

                velocity = self.velocity_compute(frame_result_1, frame_result_2, sample_ind_2 - sample_ind)

                self.velocity_result[sample_ind] = velocity

            # for i, sample_ind in enumerate(self.sample_list):
            #     pred = self.sample_result[sample_ind]['pred_boxes']
            #     if isinstance(pred, torch.Tensor):
            #         print(i, pred)

        # Initialize a cluster
        cluster_dict = self.init_label_distances(self.sample_list)

        # Check the budget remains
        remain_budget = budget - start_budget

        for i in range(remain_budget):
            rep_idx, cluster_id = self.select_rep(cluster_dict)

            self.sample_list.append(rep_idx)

            cluster_dict = self.update(cluster_dict,  cluster_id, rep_idx)

        pass

    def uniform_sample_method(self, budget):
        # frame_num = len(self.dataloader.get_file_list())
        #
        # budget = frame_num
        self.uniform_sample(budget)
        # Inference the result of all sampled frames
        for sample_ind in self.sample_list:
            data_dict = self.dataloader.get_item_ind(sample_ind)
            data_dict = self.dataloader.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = self.pred_model.forward(data_dict)
            pred_dicts = move_to_cpu(pred_dicts[0])

            # Filter out boxes with low confidence
            pred_dicts = self.filter_out_object_low_confidence(pred_dicts)

            # Store the result in the sample_result dictionary
            self.sample_result[sample_ind] = pred_dicts
            pass

        # initialize the velocity
        if self.index_mode == "velocity":
            for i, sample_ind in enumerate(self.sample_list):
                if i == len(self.sample_list) - 1:
                    break
                sample_ind_2 = self.sample_list[i + 1]
                frame_result_1 = self.sample_result[sample_ind]
                frame_result_2 = self.sample_result[sample_ind_2]

                velocity = self.velocity_compute(frame_result_1, frame_result_2, sample_ind_2 - sample_ind)

                self.velocity_result[sample_ind] = velocity


        return

    def filter_out_object_low_confidence(self, pred_dicts):
        # Filter out boxes and corresponding labels with pred_scores <= 0.5
        filtered_boxes, filtered_scores, filtered_labels = [], [], []
        for box, score, label in zip(pred_dicts['pred_boxes'], pred_dicts['pred_scores'],
                                     pred_dicts['pred_labels']):
            if score > self.pred_confidence:
                filtered_boxes.append(box)
                filtered_scores.append(score)
                filtered_labels.append(label)

        pred_dicts['pred_boxes'] = filtered_boxes
        pred_dicts['pred_scores'] = filtered_scores
        pred_dicts['pred_labels'] = filtered_labels
        return pred_dicts

    def all_sample(self):
        frame_num = int(len(self.dataloader.get_file_list()) * self.args.process_percentage)

        budget = frame_num
        self.uniform_sample(budget)
        # Inference the result of all sampled frames
        for sample_ind in self.sample_list:
            data_dict = self.dataloader.get_item_ind(sample_ind)
            data_dict = self.dataloader.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = self.pred_model.forward(data_dict)
            pred_dicts = move_to_cpu(pred_dicts[0])

            # Filter out boxes with low confidence
            pred_dicts = self.filter_out_object_low_confidence(pred_dicts)

            # Store the result in the sample_result dictionary
            self.sample_result[sample_ind] = pred_dicts


        return



    def select_rep(self, cluster_dict):

        # Initialize an empty list for indices
        rep_indices = self.sample_list


        # Compute the distances of each cluster and select based on the formula
        mab_values = []
        c_param = self.constant_param
        for cluster_key in cluster_dict:
            reward = cluster_dict[cluster_key]['distance']
            members = cluster_dict[cluster_key]['members']

            mab_value = reward + c_param * np.sqrt(2 * np.log(len(rep_indices)) / len(members))
            mab_values.append(mab_value)

        assert (len(mab_values) == len(cluster_dict.keys()))

        # Pick the one with maximum value
        selected_cluster = np.argmax(mab_values)
        start_idx, end_idx = list(cluster_dict.keys())[selected_cluster]

        # Generate the frame id for selecting
        choices = []
        for i in range(start_idx, end_idx + 1):
            if i not in cluster_dict[(start_idx, end_idx)]['members']:
                choices.append(i)

        # If no frame could be selected
        if len(choices) == 0:
            cluster_dict[(start_idx, end_idx)]['distance'] = 0
            rep_idx, (start_idx, end_idx) = self.select_rep(cluster_dict)
        else:
            rep_idx = np.random.choice(choices, 1)[0]

        return rep_idx, (start_idx, end_idx)

    def sample_sort(self):
        self.sample_list.sort()
        return

    def initiate_decision_tree(self):
        # First sort all samples
        self.sample_sort()

        # the decision tree take the sample index as input
        decision_tree = DecisionTree(self.args, self)

        return decision_tree

    def compute_predict_result(self, sample_ind):
        data_dict = self.dataloader.get_item_ind(sample_ind)
        data_dict = self.dataloader.collate_batch([data_dict])
        load_data_to_gpu(data_dict)
        pred_dicts, _ = self.pred_model.forward(data_dict)
        pred_dicts = move_to_cpu(pred_dicts[0])

        # Filter out boxes with low confidence
        pred_dicts = self.filter_out_object_low_confidence(pred_dicts)

        # # Store the result in the sample_result dictionary
        # self.sample_result[sample_ind] = pred_dicts

        return pred_dicts

    def bdm_mab_sample(self, budget, uniform_sampling_rate=0.25):
        """
        This function is the sampling method which models the procedure as a branching decision-making problem.
        Args:
            budget:
            uniform_sampling_rate:
        Returns:

        """
        # get the number of total frames
        frame_num = len(self.dataloader.get_file_list())

        # use a rate of budget to do the uniform sampling
        start_budget = max(int(budget  * uniform_sampling_rate), 2)
        
        self.uniform_sample(start_budget)

        # Inference the result of all sampled frames
        for sample_ind in self.sample_list:
            pred_dicts = self.compute_predict_result(sample_ind)

            # Store the result in the sample_result dictionary
            self.sample_result[sample_ind] = pred_dicts

        # initialize the velocity
        if self.index_mode == "velocity":
            for i, sample_ind in enumerate(self.sample_list):
                if i == len(self.sample_list) - 1:
                    break
                sample_ind_2 = self.sample_list[i + 1]
                frame_result_1 = self.sample_result[sample_ind]
                frame_result_2 = self.sample_result[sample_ind_2]

                velocity = self.velocity_compute(frame_result_1, frame_result_2, sample_ind_2 - sample_ind)

                self.velocity_result[sample_ind] = velocity

        # # Initialize a cluster
        # cluster_dict = self.init_label_distances(self.sample_list)

        decision_tree = self.initiate_decision_tree()

        # decision_tree = DecisionTree(cluster_dict)

        # Check the budget remains
        remain_budget = budget - start_budget

        for i in range(remain_budget):

            frame_id = decision_tree.sample()

            pred_dicts = self.compute_predict_result(frame_id)

            # self.sample_list.append(frame_id)

            # print("frame_id: ", frame_id)

            self.sample_list.append(frame_id)

            self.sample_result[frame_id] = pred_dicts

            pred_dicts = copy.deepcopy(pred_dicts)

            decision_tree.update(pred_dicts)

            # update the decision tree first, and then update the velocity, respectively.
            self.update_velocity(frame_id)



            # rep_idx, cluster_id = self.select_rep(cluster_dict)
            #
            # self.sample_list.append(rep_idx)
            #
            # cluster_dict = self.update(cluster_dict, cluster_id, rep_idx)

        return

