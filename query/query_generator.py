import json
import os

query_type_list = ["aggregate", "retrieval"]

object_type_list = ["car"]

spatial_sign_list = [">=", "<="]

spatial_distance_list = [2, 5, 10, 15, 20]

retrieve_filter_list = [1, 3, 5, 7, 9]

retrieve_sign_list = [">=", "<="]


def generate_query(query_dir):
    for query_type in query_type_list:

        if query_type == "aggregate":
            count = 0
            for object_type in object_type_list:
                for spatial_sign in spatial_sign_list:
                    for spatial_distance in spatial_distance_list:
                        new_query = {}
                        new_query["query_type"] = query_type
                        new_query["object_type"] = object_type
                        new_query["spatial_sign"] = spatial_sign
                        new_query["spatial_distance"] = spatial_distance
                        with open("{}/{}_{}_{}_{}.json".format(query_dir, query_type, object_type,
                                                               "leq" if spatial_sign == "<=" else "geq",
                                                               spatial_distance), 'w') as f:
                            json.dump(new_query, f)
                        count += 1
        if query_type == "retrieval":
            count = 0
            for object_type in object_type_list:
                for spatial_sign in spatial_sign_list:
                    for spatial_distance in spatial_distance_list:
                        for retrieve_sign in retrieve_sign_list:
                            for retrieve_filter in retrieve_filter_list:
                                new_query = {}
                                new_query["query_type"] = query_type
                                new_query["object_type"] = object_type
                                new_query["spatial_sign"] = spatial_sign
                                new_query["spatial_distance"] = spatial_distance
                                new_query["retrieve_sign"] = retrieve_sign
                                new_query["retrieve_filter"] = retrieve_filter
                                with open("{}/{}_{}_{}_{}_{}_{}.json".format(query_dir, query_type, object_type,
                                                                             "leq" if spatial_sign == "<=" else "geq",
                                                                             spatial_distance,
                                                                             "leq" if retrieve_sign == "<=" else "geq",
                                                                             retrieve_filter), 'w') as f:
                                    json.dump(new_query, f)
                                count += 1
    pass


if __name__ == "__main__":
    query_dir = "./query_workload"
    if not os.path.exists(query_dir):
        os.mkdir(query_dir)

    generate_query(query_dir)
