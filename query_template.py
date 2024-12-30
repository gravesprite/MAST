"""
Implement the tempolate of the analytical query
"""
import os
import json
import math
import torch
import copy
import time
import statistics

class Query:
    def __init__(self, query_type="aggregate",
                 query_parameters={"object_type": "car", "spatial_sign": "<", "spatial_distance": 10}, pred_confidence=0.5, detect_radius=75):
        self.query_type = query_type
        self.parameters = query_parameters

        self.min_pred_score = pred_confidence
        self.detect_radius = detect_radius


        if self.query_type == "aggregate":
            self.object_type = self.parameters["object_type"]
            self.spatial_sign = self.parameters["spatial_sign"]  # >= <=
            self.spatial_distance = self.parameters["spatial_distance"]  # a float number
            self.aggregate_type = self.parameters["aggregate_type"]
        if self.query_type == "retrieval":
            self.object_type = self.parameters["object_type"]
            self.retrieve_sign = self.parameters["retrieve_sign"]  # >= <=
            self.retrieve_filter = self.parameters["retrieve_filter"] # an integer
            self.spatial_sign = self.parameters["spatial_sign"]  # >= <=
            self.spatial_distance = self.parameters["spatial_distance"]  # a float number
        pass

    def compute_num(self, model_result):
        data = model_result

        # filter the boxes
        filtered_boxes = []

        pred_boxes = data['pred_boxes']
        pred_scores = data['pred_scores']

        for box, score in zip(pred_boxes, pred_scores):
            x, y = box[0].item(), box[1].item()
            # distance = math.sqrt(x ** 2 + y ** 2)
            distance = math.hypot(x, y)

            if self.spatial_sign == ">=":
                if distance >= self.spatial_distance and score.item() > self.min_pred_score and distance <= self.detect_radius:
                    filtered_boxes.append({'pred_box': box.tolist(), 'pred_score': score.item()})
            elif self.spatial_sign == "<=":
                if distance <= self.spatial_distance and score.item() > self.min_pred_score and distance <= self.detect_radius:
                    filtered_boxes.append({'pred_box': box.tolist(), 'pred_score': score.item()})

            # if distance < self.spatial_distance and score.item() > self.min_pred_score:
            #     filtered_boxes.append({'pred_box': box.tolist(), 'pred_score': score.item()})


        return len(filtered_boxes)

    def compute_num_distance_only(self, distance_situation):
        distances = distance_situation['distance']
        scores = distance_situation['score']

        count = 0
        for distance, score in zip(distances, scores):
            if self.spatial_sign == ">=":
                if distance >= self.spatial_distance and score > self.min_pred_score and distance <= self.detect_radius:
                    count += 1
            elif self.spatial_sign == "<=":
                if distance <= self.spatial_distance and score > self.min_pred_score and distance <= self.detect_radius:
                    count += 1

        # The number of elements that satisfy the condition
        num_filtered_boxes = count
        return num_filtered_boxes
        pass

    def compute_num_2(self, model_result):
        data = model_result

        # filter the boxes
        filtered_boxes = []

        pred_boxes = data['pred_boxes']
        pred_scores = data['pred_scores']

        for box, score in zip(pred_boxes, pred_scores):
            x, y = box[0].item(), box[1].item()
            print(x,y)
            distance = math.sqrt(x ** 2 + y ** 2)
            print(distance)
            if self.spatial_sign == ">=":
                if distance >= self.spatial_distance and score.item() >= self.min_pred_score and distance <= self.detect_radius:
                    filtered_boxes.append({'pred_box': box.tolist(), 'pred_score': score.item()})
            elif self.spatial_sign == "<=":
                if distance <= self.spatial_distance and score.item() >= self.min_pred_score and distance <= self.detect_radius:
                    filtered_boxes.append({'pred_box': box.tolist(), 'pred_score': score.item()})

            # if distance < self.spatial_distance and score.item() > self.min_pred_score:
            #     filtered_boxes.append({'pred_box': box.tolist(), 'pred_score': score.item()})
            print(score.item(), self.min_pred_score, score.item() >= self.min_pred_score, distance <= self.spatial_distance, distance <= self.detect_radius)

        return len(filtered_boxes)

    def update_result_with_velocity(self, pred_result, velocity, score_change):

        factor = 0.35

        velocity_frame_1 = velocity["velocity_of_box_frame_1"]
        velocity_frame_2 = velocity["additional_boxes"]

        for i in range(len(velocity_frame_1)):

            pred_result['pred_boxes'][i][:3] = pred_result['pred_boxes'][i][:3] + velocity_frame_1[i]
            if i in velocity['additional_frame_1_id']:

                pred_result['pred_scores'][i] = max(torch.tensor(0), pred_result['pred_scores'][i] - score_change * factor)

        for i in range(len(velocity_frame_2)):
            # pred_result['pred_boxes'][i + len(velocity_frame_1)][:3] = \
            #     pred_result['pred_boxes'][i + len(velocity_frame_1)][:3] + velocity_frame_2[i]
            # pred_result['pred_boxes'][i + len(velocity_frame_1)][:3] += torch.tensor(velocity_frame_2[i])
            # print(velocity_frame_1)
            # pred_result['pred_boxes'][i + len(velocity_frame_1)][:3] += velocity_frame_2[i][1]

            pred_result['pred_scores'][i + len(velocity_frame_1)] = min(torch.tensor(1), pred_result['pred_scores'][i + len(velocity_frame_1)] + score_change / factor)

        return pred_result

    def update_distance_situation(self, distance_situation, velocity, score_change):

        factor = 0.35

        distance_situation["distance"] = [ distance_situation["distance"][i] + distance_situation['distance_change'][i]
                                           for i in range(len(distance_situation["distance"]))]
        for box_id in velocity['additional_frame_1_id']:
            distance_situation['score'][box_id] = max(torch.tensor(0), distance_situation['score'][box_id] - score_change * factor)
        for box_id in range(len(velocity["velocity_of_box_frame_1"]), len(velocity["velocity_of_box_frame_1"]) + len(velocity["additional_boxes"])):
            distance_situation['score'][box_id] = min(torch.tensor(1), distance_situation['score'][box_id] + score_change / factor)

        return distance_situation

    def check_if_result_satisfy(self, num):
        if self.retrieve_sign == ">=":
            if num >= self.retrieve_filter:
                return True
            else:
                return False
        if self.retrieve_sign == "<=":
            if num <= self.retrieve_filter:
                return True
            else:
                return False
        else:
            return False

    def compute_result(self, index):

        # first sort the sampled indices
        index.sample_sort()

        num_list = []

        # retrieve the sampled results and generate approximate result
        for ind in index.sample_list:
            num_list.append(self.compute_num(index.sample_result[ind]))

        # with open('time_function_num.json', 'w') as f:
        #     json.dump(num_list, f)
        # return

        # The situation when aggregate query, linear propagation
        if self.query_type == "aggregate":

            # apply the linear propagation
            # if index.index_mode == "no_velocity" or index.index_mode == "velocity":
            if index.index_mode == "no_velocity":

                if self.aggregate_type == "Avg":
                    total_num = 0

                    for i in range(len(index.sample_list) - 1):
                        total_num += ((num_list[i] + num_list[i + 1]) / 2) * (index.sample_list[i + 1] - index.sample_list[i])

                    avg_num = total_num / (index.sample_list[-1] - index.sample_list[0])
                    return avg_num
                elif self.aggregate_type == "Max":
                    max_num = 0
                    for i in range(len(index.sample_list)):
                        if num_list[i]  > max_num:
                            max_num = num_list[i]
                    return max_num
                elif self.aggregate_type == "Med":
                    total_num_list = []
                    for i in range(len(index.sample_list) - 1):
                        total_num_list.append(num_list[i])
                        length = index.sample_list[i + 1] - index.sample_list[i]
                        num_change = (num_list[i + 1] - num_list[i]) / length
                        num = num_list[i]
                        for step in range(length - 1):
                            num = num + num_change
                            total_num_list.append(int(num))
                    total_num_list.append(num_list[len(index.sample_list) - 1])
                    med_num = statistics.median(total_num_list)
                    return med_num
                # elif self.aggregate_type == "Count":
                #
                #     pass
            elif index.index_mode == "velocity":
                if self.aggregate_type == "Avg":
                    total_num = 0

                    for i in range(len(index.sample_list) - 1):
                        total_num += ((num_list[i] + num_list[i + 1]) / 2) * (
                                    index.sample_list[i + 1] - index.sample_list[i])

                    avg_num = total_num / (index.sample_list[-1] - index.sample_list[0])
                    return avg_num

                elif self.aggregate_type == "Max":
                    # max_num = 0
                    # for i in range(len(index.sample_list)):
                    #     if num_list[i] > max_num:
                    #         max_num = num_list[i]
                    # return max_num

                    max_num = 0
                    for num in num_list:
                        if num > max_num:
                            max_num = num
                    for i in range(len(index.sample_list) - 1):
                        sample_ind = index.sample_list[i]
                        sample_ind_next = index.sample_list[i + 1]
                        for id in range(sample_ind + 1, sample_ind_next):
                            distance_situation = index.distance_situation[id]
                            num = self.compute_num_distance_only(distance_situation)
                            if num > max_num:
                                max_num = num
                    return max_num

                elif self.aggregate_type == "Med":
                    total_num_list = []
                    for i in range(len(index.sample_list) - 1):
                        total_num_list.append(num_list[i])
                        sample_ind = index.sample_list[i]
                        sample_ind_next = index.sample_list[i + 1]
                        for id in range(sample_ind + 1, sample_ind_next):
                            distance_situation = index.distance_situation[id]
                            num = self.compute_num_distance_only(distance_situation)
                            total_num_list.append(int(num))
                    total_num_list.append(num_list[len(index.sample_list) - 1])
                    med_num = statistics.median(total_num_list)
                    return med_num


            # apply the velocity propagation instead
            elif index.index_mode == "velocity":
                total_num = 0

                # change the result based on the velocity
                for i in range(len(index.sample_list) -1):

                    # add the num of the frame_1 to the total_num
                    num_frame_1 = num_list[i]
                    total_num += num_frame_1

                    # compute the approximate num of the following frames
                    sample_ind = index.sample_list[i]
                    sample_ind_next = index.sample_list[i + 1]

                    velocity = index.velocity_result[index.sample_list[i]]
                    pred_result = index.sample_result[index.sample_list[i]]

                    # Create a copy of the velocity_result and sample_result
                    # velocity = copy.deepcopy(index.velocity_result[index.sample_list[i]])
                    # pred_result = copy.deepcopy(index.sample_result[index.sample_list[i]])

                    for box in velocity["additional_boxes"]:
                        pred_result['pred_boxes'].append(box[0])
                        pred_result['pred_labels'].append(torch.tensor(box[2]))
                        # The additional box is initialized with score 0
                        pred_result['pred_scores'].append(torch.tensor(0))

                    # change the confidence score
                    score_change = 1 / (sample_ind_next - sample_ind)

                    for step in range(sample_ind_next - sample_ind - 1):

                        pred_result = self.update_result_with_velocity(pred_result, velocity, score_change)

                        num = self.compute_num(pred_result)
                        total_num += num

                        pass

                avg_num = total_num / (index.sample_list[-1] - index.sample_list[0])

                return avg_num

        # The situation when retrieval query, linear propagation
        elif self.query_type == "retrieval":
            result_list = []

            # apply the linear propagation
            if index.index_mode == "no_velocity":
                for i, num in enumerate(num_list):
                    if self.retrieve_sign == ">=" and num >= self.retrieve_filter:
                        if i == len(num_list) - 1:
                            result_list.append(index.sample_list[i])
                        else:
                            if num_list[i + 1] >= self.retrieve_filter:
                                for ind in range(index.sample_list[i], index.sample_list[i + 1]):
                                    result_list.append(ind)
                            else:
                                step = (num_list[i] - num_list[i + 1]) / (
                                        index.sample_list[i + 1] - index.sample_list[i])
                                offset = int((num_list[i] - self.retrieve_filter) / step) + 1
                                for ind in range(index.sample_list[i], index.sample_list[i] + offset):
                                    result_list.append(ind)

                    if self.retrieve_sign == "<=" and num <= self.retrieve_filter:
                        if i == len(num_list) - 1:
                            result_list.append(index.sample_list[i])
                        else:
                            if num_list[i + 1] <= self.retrieve_filter:
                                for ind in range(index.sample_list[i], index.sample_list[i + 1]):
                                    result_list.append(ind)
                            else:
                                step = (num_list[i + 1] - num_list[i]) / (
                                        index.sample_list[i + 1] - index.sample_list[i])
                                offset = int((self.retrieve_filter - num_list[i]) / step) + 1
                                for ind in range(index.sample_list[i], index.sample_list[i] + offset):
                                    result_list.append(ind)

                pass
                result = {}
                result["retrieve_count"] = len(result_list)
                result["frame_id"] = result_list
                return result

            elif index.index_mode == "velocity":

                # change the result based on the velocity
                for i in range(len(index.sample_list) - 1):

                    sample_ind = index.sample_list[i]
                    sample_ind_next = index.sample_list[i + 1]

                    num_frame_1 = num_list[i]
                    if self.check_if_result_satisfy(num_frame_1):
                        result_list.append(int(sample_ind))

                    # # If both not satisfying the number requirement, then pruning happens
                    if not self.check_if_result_satisfy(num_frame_1) and not self.check_if_result_satisfy(num_list[i + 1]):
                        continue

                    for id in range(sample_ind + 1, sample_ind_next):
                        distance_situation = index.distance_situation[id]
                        num = self.compute_num_distance_only(distance_situation)
                        if self.check_if_result_satisfy(num):
                            result_list.append(id)
                        pass

                # Check the last frame
                sample_ind_last = index.sample_list[-1]
                num_frame_last = num_list[-1]
                if self.check_if_result_satisfy(num_frame_last):
                    result_list.append(int(sample_ind_last))

                result = {}
                result["retrieve_count"] = len(result_list)
                result["frame_id"] = result_list

                return result

            elif index.index_mode == "velocity":

                # change the result based on the velocity
                for i in range(len(index.sample_list) - 1):

                    sample_ind = index.sample_list[i]
                    sample_ind_next = index.sample_list[i + 1]

                    num_frame_1 = num_list[i]
                    if self.check_if_result_satisfy(num_frame_1):
                        result_list.append(int(sample_ind))

                    # If both not satisfying the number requirement, then pruning happens
                    if not self.check_if_result_satisfy(num_frame_1) and not self.check_if_result_satisfy(num_list[i + 1]):
                        continue

                    velocity = index.velocity_result[index.sample_list[i]]
                    pred_result = index.sample_result[index.sample_list[i]]

                    # velocity = copy.deepcopy(index.velocity_result[index.sample_list[i]])
                    # pred_result = copy.deepcopy(index.sample_result[index.sample_list[i]])

                    for box in velocity["additional_boxes"]:
                        pred_result['pred_boxes'].append(box[0])
                        pred_result['pred_labels'].append(box[2])
                        # pred_result['pred_scores'].append(torch.tensor(self.min_pred_score))
                        pred_result['pred_scores'].append(torch.tensor(0))

                    # change the confidence score
                    score_change = 1 / (sample_ind_next - sample_ind)

                    total_step = 0

                    # st_time = time.time()
                    for step in range(sample_ind_next - sample_ind - 1):
                        pred_result = self.update_result_with_velocity(pred_result, velocity, score_change)

                        total_step += 1

                        num = self.compute_num(pred_result)

                        if self.check_if_result_satisfy(num):
                            result_list.append(int(sample_ind + total_step))

                            # print("frame_id {}, with predict num {}， if satisfied: {}".format(sample_ind + step, num,
                            #                                                                   self.check_if_result_satisfy(num)))
                            # self.compute_num_2(pred_result)
                            # print(pred_result)
                            # time.sleep(10)
                    # en_time = time.time()
                    # print(en_time - st_time)

                # Check the last frame
                sample_ind_last = index.sample_list[-1]
                num_frame_last = num_list[-1]
                if self.check_if_result_satisfy(num_frame_last):
                    result_list.append(int(sample_ind_last))

                result = {}
                result["retrieve_count"] = len(result_list)
                result["frame_id"] = result_list

                return result



    pass



class QueryWorkload:
    def __init__(self, workload_path):

        self.query_workload = []
        self.query_name = []
        assert(os.path.exists(workload_path))

        file_list = os.listdir(workload_path)

        for file in file_list:
            with open(workload_path + "/" + file, 'r') as f:
                query = json.load(f)

            query_type = query["query_type"]
            query_parameters = query

            query = Query(query_type, query_parameters)
            self.query_workload.append(query)
            self.query_name.append(file.split('.')[0])
        pass

    def generate_query_workload_result(self):
        pass