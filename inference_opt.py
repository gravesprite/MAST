import argparse
import glob
import os.path
from pathlib import Path
import time

import math
import json
import copy


try:
    import open3d
    from visual_utils import open3d_vis_utils as V

    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V

    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

from optimize_sample import SampleIndex
from query_template import Query, QueryWorkload

import csv


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

    # return the file list of point clouds
    def get_file_list(self):
        return self.sample_file_list

    def get_item(self, file_name):
        if self.ext == '.bin':
            points = np.fromfile(file_name, dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(file_name)
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': self.sample_file_list.index(file_name),
        }

    def get_item_ind(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    # Add the parameters
    parser.add_argument("--decision_tree_max_depth", type=int, default=10, help='the max depth of the decision tree')
    parser.add_argument("--c_para", type=float, default=2,
                        help='The constant parameter that compute the value of mab')
    parser.add_argument("--detect_radius", type=float, default=75.0, help='The radius of the object detection')
    parser.add_argument("--sampling_method", type=str, default='ma_mab', help='The selection of sampling method')
    parser.add_argument("--predict_method", type=str, default='velocity',
                        help='The predict method, velocity or no-velocity')
    parser.add_argument("--move_distance_ratio", type=float, default=1,
                        help='The distance move of the not appeared object')
    parser.add_argument("--generate_gt", action='store_true', help="whether compute the ground truth result or not")
    parser.add_argument("--budget_ratio", type=float, default=0.1, help='The budget of deep model sampling')
    parser.add_argument("--uniform_sampling_budget_ratio", type=float, default=0.05,
                        help='The budget of deep model sampling')
    parser.add_argument("--sequence_id", type=str, default='01', help='The  id of experiment sequence')

    # The hyperparameters
    parser.add_argument("--reward_number_factor", type=float, default=0.0, help='The factor of vary number reward')

    parser.add_argument("--process_percentage", type=float, default=1.0, help='processing percentage of the dataset')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def generate_csv(total_frame, retrieve_result, sample_list):
    path = ''

    rows = []
    retrieve_result = retrieve_result["frame_id"]

    for frame_id in range(total_frame):
        if frame_id in sample_list:
            if frame_id in retrieve_result:
                label = True
                score = 1
            else:
                label = False
                score = 0
        else:
            if frame_id in retrieve_result:
                label = None
                score = 0.9
            else:
                label = None
                score = 0.1

        row = [frame_id, label, score]
        rows.append(row)

    # Define the filename for the CSV file
    filename = 'proxy_result.csv'
    column_name = ['id', 'label', 'proxy_score']
    # Write rows to CSV file
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(column_name)
        csvwriter.writerows(rows)

    return filename


def summarize_experiment_result(exp_result):
    keys = exp_result.keys()

    count = 0
    score = 0

    for key in keys:
        if key.startswith('retrieval'):
            result = exp_result[key]
            if 'information' in result.keys():
                continue
            else:
                f1 = result['f1']
                score += f1
                count += 1

    avg_score = score / count
    print('avg_score', avg_score)

    sum_acc = 0
    count = 0

    for key in keys:
        if key.startswith('aggregate'):
            result = exp_result[key]
            accuracy = result['accuracy']
            sum_acc += accuracy
            count += 1
    avg_acc = sum_acc / count
    print('avg_agg_accuracy: ', avg_acc)

    pass


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger("logger.txt")

    logger.info('-----------------START PC ANALYTICAL EXPERIMENT-------------------------')

    if args.data_path == 'once':
        data_path = '/root/sda4/data/once/data/{}/lidar_roof'.format(args.sequence_id)
    elif args.data_path == 'semantic_kitti':
        data_path = '../data/semantic_kitti/dataset/sequences/{}/velodyne'.format(args.sequence_id)
    elif args.data_path == 'synlidar':
        data_path = '/root/sda4/data/synlidar/{}/velodyne'.format(args.sequence_id)

    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(data_path), ext=args.ext, logger=logger
    )

    logger.info(f'Total number of samples: \t{int(len(demo_dataset) * args.process_percentage)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()



    q_workload = QueryWorkload("./query/query_workload")
    query_list = q_workload.query_workload

    # sample_query =query_list[0]
    query_ind = q_workload.query_name.index("retrieval_car_geq_5_leq_1")
    sample_query = query_list[query_ind]

    experiment_results = {}

    with torch.no_grad():
        # Construct the index, index_mode whether "no_velocity" or "velocity"
        index = SampleIndex(args, model, demo_dataset, sample_query, index_mode=args.predict_method)

        if args.generate_gt == True:

            logger.info("Start generate the ground truth of {}".format(args.data_path))

            all_st_time = time.time()
            # Set the index mode and sample all
            index.index_mode = "no_velocity"
            index.all_sample()
            all_end_time = time.time()
            logger.info("all sample time usage: {} sec".format(all_end_time - all_st_time))


            # return
            total_query_time_gt = 0
            for i, query in enumerate(query_list):


                q_s_time = time.time()
                query_result = query.compute_result(index)
                q_e_time = time.time()
                total_query_time_gt += q_e_time - q_s_time

                # If the directory does not exist, new a directory
                if not os.path.exists(("./query/query_result_gt/{}".format(args.data_path))):
                    os.mkdir(
                        "./query/query_result_gt/{}".format(args.data_path))
                if not os.path.exists(
                        ("./query/query_result_gt/{}/{}".format(args.data_path, args.sequence_id))):
                    os.mkdir(
                        "./query/query_result_gt/{}/{}".format(args.data_path, args.sequence_id))

                with open("./query/query_result_gt/{}/{}/{}_result_{}.json".format(args.data_path, args.sequence_id, q_workload.query_name[i],
                                                                                   args.process_percentage),
                          'w') as f:
                    json.dump(query_result, f)

            logger.info("Generating the ground truth of {} dataset ends.".format(args.data_path))
            logger.info(" Total query time {} sec".format(total_query_time_gt))
            return

        # Conduct the sampling procedure

        budget = int(demo_dataset.__len__() * args.budget_ratio * args.process_percentage)
        logger.info("==> Sampling frame budget: {}".format(budget))

        sample_st_time = time.time()
        if args.sampling_method == "mab":
            index.mab_sample(budget)
            # index.bdm_mab_sample(400)
        elif args.sampling_method == "ma_mab":
            index.bdm_mab_sample(budget, args.uniform_sampling_budget_ratio)

        sample_end_time = time.time()
        logger.info("sample finish time: {} sec".format(sample_end_time - sample_st_time))


        logger.info("==> Sampling finished.")

        if index.index_mode == 'velocity':
            dis_st_time = time.time()
            index.generate_distance()
            index.predict_all_distance()
            dis_end_time = time.time()

            logger.info("Generate index time {} sec".format(dis_end_time - dis_st_time))

        sample_result = index.sample_result
        velocity_result = index.velocity_result
        distance_situation = index.distance_situation


        total_query_time = 0

        for i, query in enumerate(query_list):

            index.sample_result = copy.deepcopy(sample_result)
            index.velocity_result = copy.deepcopy(velocity_result)
            index.distance_situation = copy.deepcopy(distance_situation)

            query_st_time = time.time()
            query_result = query.compute_result(index)
            query_end_time = time.time()
            total_query_time += query_end_time - query_st_time


            gt_result_file = './query/query_result_gt/{}/{}/'.format(args.data_path, args.sequence_id) + q_workload.query_name[
                i] + "_result_{}.json".format(args.process_percentage)
            with open(gt_result_file, 'r') as f:
                gt_result = json.load(f)

            """
            Exam the correctness of the query results
            """
            if query.query_type == "aggregate":

                res = abs(query_result - gt_result)
                if gt_result == 0:
                    if res == 0:
                        accuracy = 100
                    else:
                        accuracy = 0
                else:
                    error = res / (gt_result)
                    accuracy = (1 - error) * 100
                experiment_results[q_workload.query_name[i]] = {}
                experiment_results[q_workload.query_name[i]]['query_result'] = query_result
                experiment_results[q_workload.query_name[i]]['gt_result'] = gt_result
                experiment_results[q_workload.query_name[i]]['accuracy'] = accuracy
                experiment_results[q_workload.query_name[i]]['time'] = query_end_time - query_st_time
            elif query.query_type == "retrieval":
                experiment_results[q_workload.query_name[i]] = {}

                # The situation when the retrieval ground truth is zero
                if gt_result['retrieve_count'] == 0:
                    experiment_results[q_workload.query_name[i]]['information'] = 'no ground truth'
                    continue
                elif query_result['retrieve_count'] == 0:
                    recall = 0
                    precision = 0
                    F1 = 0
                else:
                    count = 0
                    for ind in query_result['frame_id']:
                        if ind in gt_result['frame_id']:
                            count += 1
                    precision = count / gt_result['retrieve_count']
                    recall = count / query_result['retrieve_count']

                    if (precision + recall == 0):
                        F1 = 0
                    else:
                        F1 = 2 * ((precision * recall) / (precision + recall))

                experiment_results[q_workload.query_name[i]]['retrieve_count'] = query_result['retrieve_count']
                experiment_results[q_workload.query_name[i]]['gt_count'] = gt_result['retrieve_count']
                experiment_results[q_workload.query_name[i]]['f1'] = F1
                experiment_results[q_workload.query_name[i]]['recall'] = recall
                experiment_results[q_workload.query_name[i]]['precision'] = precision
                experiment_results[q_workload.query_name[i]]['time'] = query_end_time - query_st_time

        # logger.info(" Total query time {} sec".format(total_query_time))

        # Sorted dictionary using dictionary comprehension
        experiment_results = {k: experiment_results[k] for k in sorted(experiment_results)}

        # If the directory does not exist, new a directory
        if not os.path.exists("./query/experiment_result"):
            os.mkdir("./query/experiment_result")
        if not os.path.exists("./query/experiment_result/{}".format(args.data_path)):
            os.mkdir("./query/experiment_result/{}".format(args.data_path))
        if not os.path.exists("./query/experiment_result/{}/{}".format(args.data_path, args.sequence_id)):
            os.mkdir("./query/experiment_result/{}/{}".format(args.data_path, args.sequence_id))

        with open('./query/experiment_result/{}/{}/{}_{}_{}_{}.json'.format(args.data_path,
                                                                    args.sequence_id, args.sampling_method,
                                                                    args.predict_method, args.budget_ratio,
                                                                            args.process_percentage), 'w') as f:
            json.dump(experiment_results, f, indent=4)

        # # Generate the average f1 score
        summarize_experiment_result(experiment_results)




    logger.info("total query time: {} seconds".format(total_query_time))

    return


if __name__ == '__main__':
    main()
