import subprocess
from itertools import product

lambda_rate = [2, 4, 6, 8, 10]
cost_diff = [1.2, 1.4, 1.6, 1.8, 2.0]
func_num = [10, 20, 30, 40, 50]
cpu_core = [100, 200, 300, 400, 500]
cache_capacity = [256, 512, 768, 1024]
soft_ddl_param = [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]]

node_selection_strategy = [
    "NS_min_dis_own",
    "NS_min_dis_all",
    "NS_min_user",
    "NS_max_node",
    "NS_exist_func",
]
path_selection_strategy = ["PS_min_dis", "PS_max_cache", "PS_max_cpu"]
task_sorting_strategy = [
    "TS_exec_time_asc",
    "TS_data_vol_asc",
    "TS_exec_time_to_data_vol_ratio_asc",
    "TS_closest_soft_ddl",
    "TS_highest_response_ratio",
]

# 生成所有可能的组合
combinations = list(
    product(
        lambda_rate,
        cost_diff,
        func_num,
        cpu_core,
        cache_capacity,
        soft_ddl_param,
        node_selection_strategy,
        path_selection_strategy,
        task_sorting_strategy,
    ))

subprocess.run(["python", "AIHO-Plus.py", *combinations[-1]])
