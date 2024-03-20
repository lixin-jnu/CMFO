import numpy as np
import pandas as pd
import networkx as nx
import math, copy, pickle
from itertools import product


##################################
# ----------各种工具类----------- #
##################################
# 1.任务类
class Task:
    def __init__(self, taskId, arvTime, execTime, softDdl, dataVol, cpuCore,
                 instNum, funcId, serviceId):
        self.taskId = taskId  #任务Id
        self.arvTime = arvTime  #到达时间
        self.execTime = execTime  #执行时间
        self.softDdl = softDdl  #软截止期
        self.dataVol = dataVol  #数据量
        self.cpuCore = cpuCore  #Cpu核心数
        self.instNum = instNum  #实例个数
        self.funcId = funcId  #函数Id
        self.serviceId = serviceId  #服务商Id


# 2.边缘节点类
class Node:
    def __init__(self, nodeId, serviceId, cpuCore, cacheCapacity, funcSet,
                 bandwidth, recv_queue, wait_queue, exec_queue, tran_queue):
        self.nodeId = nodeId  #节点Id
        self.serviceId = serviceId  #服务商Id
        self.cpuCore = cpuCore  #Cpu核心数
        self.cacheCapacity = cacheCapacity  #缓存容量
        self.funcSet = funcSet  #函数集合
        self.bandwidth = bandwidth  #固定带宽
        self.recv_queue = recv_queue  #接收队列:接收用户传输过来的任务
        self.wait_queue = wait_queue  #等待队列:前一时隙累积+本时隙Local+本时隙卸载而来+本时隙冷启动完成
        self.exec_queue = exec_queue  #执行队列:真正分配资源开始执行
        self.tran_queue = tran_queue  #传输队列:接收其它边缘节点传输过来的任务+本节点冷启动的任务


##################################
# ---------各种工具函数---------- #
##################################
# 1.从func中按比例随机生成本次实验的函数集
def getFuncSet(func_num):
    net = getFuncSetByType(".net", func_num)
    golang = getFuncSetByType("golang", func_num)
    python = getFuncSetByType("python", func_num)
    nodejs = getFuncSetByType("node.js", func_num)
    java = getFuncSetByType("java", func_num)
    php = getFuncSetByType("php", func_num)
    res = np.vstack((net, golang, python, nodejs, java, php))
    return np.delete(res, np.s_[2:2 + len(res) - func_num], axis=0)


def getFuncSetByType(runc_type, func_num):
    n = len(func)
    subFunc = func[func["runc_type"] == runc_type]
    m = len(subFunc)
    num = int(np.round(func_num * (m / n)))
    num = num if num >= 1 else 1
    return subFunc.sample(n=num, replace=False).to_numpy()


##################################
# ----------AIHO-Plus----------- #
##################################
def AIHO_Plus(lambda_rate, func_num, cost_diff, cpu_core, cache_capacity,
              soft_ddl_param, node_selection_strategy, path_selection_strategy,
              task_sorting_strategy):

    # ===初始化函数集和软截止期===
    n = 125
    bandwidth_sub_1 = 1000.0
    bandwidth_sub_6 = 2000.0
    bandwidth_mmWave_24 = 4000.0
    func_set = getFuncSet(func_num)  # ["python", 1.0, 512.0]
    task["soft_ddl"] = task["exec_time"].apply(lambda x: math.ceil(
        max(
            1.0,  # [0.1, 0.2)
            np.random.uniform(soft_ddl_param[0], soft_ddl_param[1]) * x)))

    # ===初始化每个基站的CPU核心数|缓存容量|函数集合|网络带宽===
    sub_1 = np.array([
        30, 89, 52, 57, 93, 53, 1, 56, 12, 116, 38, 50, 86, 76, 84, 28, 22, 98
    ])
    sub_6 = np.array([
        103, 54, 108, 45, 95, 36, 58, 14, 6, 11, 0, 96, 88, 25, 104, 113, 80,
        81, 94, 7, 31, 92, 115, 39, 21, 72, 4, 9, 105, 85, 37, 63, 107, 17, 79,
        26
    ])
    mmWave_24 = np.array([
        32, 55, 124, 118, 101, 3, 106, 97, 66, 19, 68, 13, 122, 70, 23, 59, 15,
        47, 10, 87, 102, 65, 29, 2, 73, 5, 75, 114, 117, 77, 33, 35, 60, 8, 46,
        111, 112, 71, 43, 61, 27, 51, 69, 121, 90, 16, 41, 64, 67, 34, 99, 120,
        123, 62, 100, 74, 109, 42, 119, 48, 78, 24, 83, 110, 20, 44, 91, 40,
        82, 18, 49
    ])
    node_list = []
    for i in np.arange(n):
        if i in sub_1:
            node_list.append(
                Node(
                    nodeId=i,
                    serviceId=1,
                    cpuCore=cpu_core,
                    cacheCapacity=cache_capacity * 1024.0,
                    funcSet={},
                    bandwidth=bandwidth_sub_1,
                    recv_queue=[],
                    wait_queue=[],
                    exec_queue=[],
                    tran_queue=[],
                ))
        elif i in sub_6:
            node_list.append(
                Node(
                    nodeId=i,
                    serviceId=6,
                    cpuCore=cpu_core,
                    cacheCapacity=cache_capacity * 1024.0,
                    funcSet={},
                    bandwidth=bandwidth_sub_6,
                    recv_queue=[],
                    wait_queue=[],
                    exec_queue=[],
                    tran_queue=[],
                ), )
        else:  # i in mmWave_24
            node_list.append(
                Node(
                    nodeId=i,
                    serviceId=24,
                    cpuCore=cpu_core,
                    cacheCapacity=cache_capacity * 1024.0,
                    funcSet={},
                    bandwidth=bandwidth_mmWave_24,
                    recv_queue=[],
                    wait_queue=[],
                    exec_queue=[],
                    tran_queue=[],
                ))
    node_list = np.array(node_list)

    # ===初始化新的Sub_1|Sub_6|mmWave_24|Cross_net基站拓扑图===
    G_sub_1 = copy.deepcopy(graph["sub-1"])
    G_sub_6 = copy.deepcopy(graph["sub-6"])
    G_mmWave_24 = copy.deepcopy(graph["mmWave-24"])
    G_cross_net = copy.deepcopy(graph["cross-net"])

    for u, v, _ in G_sub_1.edges(data=True):
        G_sub_1[u][v]["weight"] = 1.0 / bandwidth_sub_1

    for u, v, _ in G_sub_6.edges(data=True):
        G_sub_6[u][v]["weight"] = 1.0 / bandwidth_sub_6

    for u, v, _ in G_mmWave_24.edges(data=True):
        G_mmWave_24[u][v]["weight"] = 1.0 / bandwidth_mmWave_24

    for u, v, _ in G_cross_net.edges(data=True):
        if u in sub_1:
            bandwidth = bandwidth_sub_1
        elif u in sub_6:
            bandwidth = bandwidth_sub_6
        else:
            bandwidth = bandwidth_mmWave_24
        if ((u in sub_1 and v in sub_1) or (u in sub_6 and v in sub_6)
                or (u in mmWave_24 and v in mmWave_24)):
            G_cross_net[u][v]["weight"] = 1.0 / bandwidth
        else:
            G_cross_net[u][v]["weight"] = cost_diff / bandwidth

    G = {}
    G[1] = G_sub_1
    G[6] = G_sub_6
    G[24] = G_mmWave_24
    G["cross-net"] = G_cross_net

    ##################################
    # ---------节点选择策略---------- #
    ##################################
    # 1.min_dis_own:选择距离最近的本服务提供商的基站
    def NS_min_dis_own(userId, funcId=-1):
        serviceId = 1 if userId % 3 == 1 else (6 if userId % 3 == 2 else 24)
        sortedNdLst = sorted(user[userId], key=lambda x: x[2])
        for nd in sortedNdLst:
            if nd[1] == serviceId:
                return nd[0]

    # 2.min_dis_all:选择距离最近的基站
    def NS_min_dis_all(userId, funcId=-1):
        return min(user[userId], key=lambda x: x[2])[0]

    # 3.min_user:选择覆盖用户数最少的基站
    def NS_min_user(userId, funcId=-1):
        return min(user[userId], key=lambda x: node[x[0]][0])[0]

    # 4.max_node:选择邻居基站数最多的基站
    def NS_max_node(userId, funcId=-1):
        return max(user[userId], key=lambda x: node[x[0]][1])[0]

    # 5.max_cpu:选择负载最低(剩余CPU核心数最多)的基站
    def NS_max_cpu(userId, funcId=-1):
        return max(user[userId], key=lambda x: node_list[x[0]].cpuCore)[0]

    # 6.max_cache:选择剩余缓存容量最多的基站
    def NS_max_cache(userId, funcId=-1):
        return max(user[userId],
                   key=lambda x: node_list[x[0]].cacheCapacity)[0]

    # 7.exist_func:选择距离最近的存在函数的基站
    def NS_exist_func(userId, funcId):
        sortedNdLst = sorted(user[userId], key=lambda x: x[2])
        for nd in sortedNdLst:
            if funcId in node_list[nd[0]].funcSet:
                return nd[0]
        # 均不存在函数时退化为min_dis_all策略
        return sortedNdLst[0][0]

    NS = {
        "NS_min_dis_own": NS_min_dis_own,
        "NS_min_dis_all": NS_min_dis_all,
        "NS_min_user": NS_min_user,
        "NS_max_node": NS_max_node,
        "NS_max_cpu": NS_max_cpu,
        "NS_max_cache": NS_max_cache,
        "NS_exist_func": NS_exist_func
    }

    ##################################
    # ---------路径选择策略---------- #
    ##################################
    # 1.min_dis:寻找可以处理任务task的距离节点nodeId最近的边缘节点
    def PS_min_dis(G, nodeId, tk):
        # 按照距离边缘节点nodeId的最短路径大小正序排序
        for tarNd, count in nx.single_source_dijkstra_path_length(
                G, nodeId).items():
            if tarNd == nodeId:
                continue
            curNd = node_list[tarNd]
            if (tk.instNum * tk.cpuCore <= curNd.cpuCore) and (
                    tk.instNum * func_set[tk.funcId][2] <=
                    curNd.cacheCapacity) and (tk.funcId in curNd.funcSet):
                return tarNd, tk.dataVol * count
        return -1, np.finfo(np.float32).max

    # 2.max_cpu:寻找可以处理任务task的剩余Cpu核心数最多的边缘节点
    def PS_max_cpu(G, nodeId, tk):
        # 按照剩余Cpu核心数倒序排序,如果相同按照距离边缘节点nodeId的最短路径大小正序排序
        sortedNdlst = sorted(
            G.nodes(),
            key=lambda x:
            (-node_list[x].cpuCore,
             nx.shortest_path_length(G, nodeId, x, weight="weight")))
        for tarNd in sortedNdlst:
            if tarNd == nodeId:
                continue
            curNd = node_list[tarNd]
            if (tk.instNum * tk.cpuCore <= curNd.cpuCore) and (
                    tk.instNum * func_set[tk.funcId][2] <=
                    curNd.cacheCapacity) and (tk.funcId in curNd.funcSet):
                return tarNd, tk.dataVol * nx.shortest_path_length(
                    G, nodeId, tarNd, weight="weight")
        return -1, np.finfo(np.float32).max

    # 3.max_cache:寻找可以处理任务task的剩余缓存容量最多的边缘节点
    def PS_max_cache(G, nodeId, tk):
        # 按照剩余缓存容量倒序排序,如果相同按照距离边缘节点nodeId的最短路径大小正序排序
        sortedNdlst = sorted(
            G.nodes(),
            key=lambda x:
            (-node_list[x].cacheCapacity,
             nx.shortest_path_length(G, nodeId, x, weight="weight")))
        for tarNd in sortedNdlst:
            if tarNd == nodeId:
                continue
            curNd = node_list[tarNd]
            if (tk.instNum * tk.cpuCore <= curNd.cpuCore) and (
                    tk.instNum * func_set[tk.funcId][2] <=
                    curNd.cacheCapacity) and (tk.funcId in curNd.funcSet):
                return tarNd, tk.dataVol * nx.shortest_path_length(
                    G, nodeId, tarNd, weight="weight")
        return -1, np.finfo(np.float32).max

    PS = {
        "PS_min_dis": PS_min_dis,
        "PS_max_cache": PS_max_cache,
        "PS_max_cpu": PS_max_cpu
    }

    ##################################
    # ---------任务排序策略---------- #
    ##################################
    # 1.exec_time_asc:按照任务的执行时间升序排序
    def TS_exec_time_asc(wait_queue, cur_time):
        return sorted(wait_queue, key=lambda x: x.execTime)

    # 2.exec_time_desc:按照任务的执行时间降序排序
    def TS_exec_time_desc(wait_queue, cur_time):
        return sorted(wait_queue, key=lambda x: -x.execTime)

    # 3.data_vol_asc:按照任务的数据量升序排序
    def TS_data_vol_asc(wait_queue, cur_time):
        return sorted(wait_queue, key=lambda x: x.dataVol)

    # 4.data_vol_desc:按照任务的数据量降序排序
    def TS_data_vol_desc(wait_queue, cur_time):
        return sorted(wait_queue, key=lambda x: -x.dataVol)

    # 5.exec_time_to_data_vol_ratio_asc:按照任务的执行时间/数据量升序排序
    def TS_exec_time_to_data_vol_ratio_asc(wait_queue, cur_time):
        return sorted(wait_queue, key=lambda x: x.execTime / x.dataVol)

    # 6.exec_time_to_data_vol_ratio_desc:按照任务的执行时间/数据量降序排序
    def TS_exec_time_to_data_vol_ratio_desc(wait_queue, cur_time):
        return sorted(wait_queue, key=lambda x: -x.execTime / x.dataVol)

    # 7.closest_soft_ddl:按照任务的软截止期逼近程度排序
    def TS_closest_soft_ddl(wait_queue, cur_time):
        return sorted(
            wait_queue,
            key=lambda x: x.arvTime + x.execTime + x.softDdl - cur_time)

    # 8.highest_response_ratio:按照任务的响应比降序排序
    def TS_highest_response_ratio(wait_queue, cur_time):
        return sorted(
            wait_queue,
            key=lambda x: -(x.execTime + cur_time - x.arvTime) / x.execTime)

    TS = {
        "TS_exec_time_asc": TS_exec_time_asc,
        "TS_exec_time_desc": TS_exec_time_desc,
        "TS_data_vol_asc": TS_data_vol_asc,
        "TS_data_vol_desc": TS_data_vol_desc,
        "TS_exec_time_to_data_vol_ratio_asc":
        TS_exec_time_to_data_vol_ratio_asc,
        "TS_exec_time_to_data_vol_ratio_desc":
        TS_exec_time_to_data_vol_ratio_desc,
        "TS_closest_soft_ddl": TS_closest_soft_ddl,
        "TS_highest_response_ratio": TS_highest_response_ratio,
    }

    task_num = len(task)  #任务总数
    task_over_num = 0  #已完成任务
    task_use_num = 0  #当前使用到数据集中的第几个任务|下次拿任务从第task_use_num个获取

    t = 0  #当前时隙

    # 采用不同处理方式的任务个数
    Local = 0
    Inter_Edge = 0
    Cross_Edge = 0
    Cold_Start = 0

    # 用于任务延迟时间的计算
    arrivalDic = {}
    delayDic = {}

    # 在每一时隙开始时进行调度
    while task_over_num < task_num:

        # 边缘节点不同队列的数据格式
        # recv_queue|tram_queue->[[task,t],...]
        # wait_queue|exec_queue->[task,...]

        # 在新时隙开始时对每个边缘节点的接收队列+传输队列+执行队列中的时间都减1
        for nd in np.arange(n):
            curNd = node_list[nd]
            RecvQe = curNd.recv_queue
            TranQe = curNd.tran_queue
            ExecQe = curNd.exec_queue
            for i in range(len(RecvQe) - 1, -1, -1):
                RecvQe[i][1] -= 1
            for i in range(len(TranQe) - 1, -1, -1):
                TranQe[i][1] -= 1
            for i in range(len(ExecQe) - 1, -1, -1):
                ExecQe[i].execTime -= 1
                # 不仅执行时间-1,还要恢复Cpu核心数+缓存容量+删除任务+删除函数
                if ExecQe[i].execTime == 0:
                    curNd.funcSet[ExecQe[i].funcId] -= 1
                    if curNd.funcSet[ExecQe[i].funcId] == 0:
                        # del curNd.funcSet[ExecQe[i].funcId]
                        curNd.funcSet.pop(ExecQe[i].funcId)
                    curNd.cpuCore += ExecQe[i].instNum * ExecQe[i].cpuCore
                    curNd.cacheCapacity += ExecQe[i].instNum * func_set[
                        ExecQe[i].funcId][2]
                    task_over_num += 1
                    delayDic[ExecQe[i].taskId] = max(
                        t - arrivalDic[ExecQe[i].taskId], 0)
                    del ExecQe[i]

        # 依旧存在没有到来的任务
        if task_use_num < task_num:
            # 每个用户开始生成任务
            for u in user:
                if task_use_num >= task_num:
                    break
                # 该用户在t时隙生成的任务个数
                utNum = np.random.poisson(lambda_rate)
                for i in np.arange(utNum):
                    if task_use_num >= task_num:
                        break
                    # 构建任务对象
                    tk = Task(
                        taskId=task_use_num,
                        arvTime=t,
                        execTime=task.iloc[task_use_num]["exec_time"],
                        softDdl=task.iloc[task_use_num]["soft_ddl"],
                        dataVol=task.iloc[task_use_num]["data_vol"],
                        cpuCore=task.iloc[task_use_num]["plan_cpu"],
                        instNum=task.iloc[task_use_num]["inst_num"],
                        funcId=np.random.randint(0, func_num),
                        serviceId=1 if u % 3 == 1 else
                        (6 if u % 3 == 2 else 24),
                    )
                    # 记录每个任务最晚应该完成的时隙
                    arrivalDic[tk.taskId] = t + tk.execTime + tk.softDdl
                    '''
                    #-----------------#
                    #   节点选择组件   #
                    #-----------------#
                    '''
                    selectedNode = NS[node_selection_strategy](u, tk.funcId)
                    #------------------#
                    #    Recv_Queue    #
                    #------------------#
                    node_list[selectedNode].recv_queue.append([
                        tk,
                        math.ceil(tk.dataVol /
                                  (node_list[selectedNode].bandwidth /
                                   node[selectedNode][0]))  #数据量/(基站带宽/基站覆盖用户数)
                    ])
                    task_use_num += 1
        else:
            # 在全部任务到来后结束本次实验(暂时)
            break

        # 对每个边缘节点接收到的到期任务进行卸载决策并放入到对应边缘节点的等待队列或传输队列
        for nd in np.arange(n):
            curNd = node_list[nd]
            RecvQe = curNd.recv_queue
            for i in range(len(RecvQe) - 1, -1, -1):
                # 任务tk到达边缘节点nd
                if RecvQe[i][1] == 0:
                    tk = RecvQe[i][0]
                    del RecvQe[i]
                    '''
                    #-----------------#
                    #   卸载决策组件   #
                    #-----------------#
                    '''
                    # Local:当前边缘节点即可处理该任务(Cpu+缓存+存在函数)
                    if (tk.instNum * tk.cpuCore <= curNd.cpuCore) and (
                            tk.instNum * func_set[tk.funcId][2] <=
                            curNd.cacheCapacity) and (tk.funcId
                                                      in curNd.funcSet):
                        #------------------#
                        #    Wait_Queue    #
                        #------------------#
                        curNd.wait_queue.append(tk)
                        # Local方式延迟时间的计算
                        # t=结束时隙+基站将任务传回给用户时间-开始时隙-任务执行时间-任务软截止期时间
                        # t=结束时隙-(开始时隙+任务执行时间+任务软截止期时间-基站将任务传回给用户时间)
                        arrivalDic[tk.taskId] -= math.ceil(
                            (tk.dataVol / 10) /
                            (curNd.bandwidth / node[nd][0]))  #数据量/10
                        Local += 1
                        continue
                    '''
                    #-----------------#
                    #   路径选择组件   #
                    #-----------------#
                    '''
                    # Inter-Edge:在该基站的内网寻找其它基站处理该任务
                    ieNd, ieCost = PS[path_selection_strategy](
                        G[curNd.serviceId], nd, tk)
                    # Cross-Edge:在全网寻找其它基站处理该任务
                    ceNd, ceCost = PS[path_selection_strategy](G["cross-net"],
                                                               nd, tk)
                    # Cold-start:在当前边缘节点冷启动处理该任务
                    csNd, csCost = nd, func_set[tk.funcId][1]
                    # 选择代价最小的处理方式
                    offload_decision = min(
                        [[0, ieNd, ieCost], [1, ceNd, ceCost],
                         [2, csNd, csCost]],
                        key=lambda x: x[2])
                    # Inter-Edge方式延迟时间的计算
                    # t=结束时隙+基站将任务传回到达基站的时间+到达基站将任务传回给用户时间-开始时隙-任务执行时间-任务软截止期时间
                    # t=结束时隙-(开始时隙+任务执行时间+任务软截止期时间-基站将任务传回到达基站的时间-到达基站将任务传回给用户时间)
                    if offload_decision[0] == 0:
                        arrivalDic[tk.taskId] -= math.ceil((
                            (tk.dataVol / 10) * nx.shortest_path_length(
                                G[curNd.serviceId], ieNd, nd, weight="weight")
                        ) + ((tk.dataVol / 10) /
                             (curNd.bandwidth / node[nd][0])))
                        Inter_Edge += 1
                    # Cross-Edge方式延迟时间的计算
                    elif offload_decision[0] == 1:
                        arrivalDic[tk.taskId] -= math.ceil((
                            (tk.dataVol / 10) * nx.shortest_path_length(
                                G["cross-net"], ceNd, nd, weight="weight")) + (
                                    (tk.dataVol / 10) /
                                    (curNd.bandwidth / node[nd][0])))
                        Cross_Edge += 1
                    # Cold-start方式延迟时间的计算
                    else:  #同Local方式
                        arrivalDic[tk.taskId] -= math.ceil(
                            (tk.dataVol / 10) /
                            (curNd.bandwidth / node[nd][0]))
                        Cold_Start += 1
                    #------------------#
                    #    Tran_Queue    #
                    #------------------#
                    node_list[offload_decision[1]].tran_queue.append(
                        [tk, math.ceil(offload_decision[2])])

        # 将每个边缘节点卸载而来的任务+冷启动完成的任务放入执行队列
        for nd in np.arange(n):
            curNd = node_list[nd]
            TranQe = curNd.tran_queue
            for i in range(len(TranQe) - 1, -1, -1):
                if TranQe[i][1] == 0:
                    tk = TranQe[i][0]
                    del TranQe[i]
                    #------------------#
                    #    Wait_Queue    #
                    #------------------#
                    curNd.wait_queue.append(tk)

        # 对每个边缘节点等待队列中的任务进行任务排序并依次放入执行队列
        for nd in np.arange(n):
            curNd = node_list[nd]
            '''
            #-----------------#
            #   任务排序组件   #
            #-----------------#
            '''
            curNd.wait_queue = TS[task_sorting_strategy](curNd.wait_queue, t)
            WaitQe = curNd.wait_queue
            # 开始进行真正的资源分配
            for i in range(len(WaitQe) - 1, -1, -1):
                tk = WaitQe[i]
                if tk.instNum * tk.cpuCore <= curNd.cpuCore and (
                        tk.instNum * func_set[tk.funcId][2] <=
                        curNd.cacheCapacity):
                    #------------------#
                    #    Exec_Queue    #
                    #------------------#
                    curNd.exec_queue.append(tk)
                    curNd.funcSet[tk.funcId] = curNd.funcSet.get(tk.funcId,
                                                                 0) + 1
                    curNd.cpuCore -= tk.instNum * tk.cpuCore
                    curNd.cacheCapacity -= tk.instNum * func_set[tk.funcId][2]
                    del WaitQe[i]

        # print("===================")
        # print(task_use_num)
        # print(task_over_num)
        # print(t, Local, Inter_Edge, Cross_Edge, Cold_Start)
        # print("===================")
        t += 1

    return np.mean(list(delayDic.values()))


# ===读取数据集===
task = pd.read_pickle(r"outData/task-2.pkl")
func = pd.read_pickle(r"outData/func.pkl")
with open(r"outData/graph.pkl", "rb") as file:
    graph = pickle.load(file)
with open(r"outData/user.pkl", "rb") as file:
    user = pickle.load(file)
with open(r"outData/node.pkl", "rb") as file:
    node = pickle.load(file)

lambda_rate_sp = [2, 4, 6, 8, 10]
cost_diff_sp = [1.2, 1.4, 1.6, 1.8, 2.0]
func_num_sp = [10, 20, 30, 40, 50]
cpu_core_sp = [100, 200, 300, 400, 500]
cache_capacity_sp = [256, 512, 768, 1024]
soft_ddl_param_sp = [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]]

node_selection_strategy_sp = [
    "NS_min_dis_own",
    "NS_min_dis_all",
    "NS_min_user",
    "NS_max_node",
    "NS_exist_func",
]
path_selection_strategy_sp = ["PS_min_dis", "PS_max_cache", "PS_max_cpu"]
task_sorting_strategy_sp = [
    "TS_exec_time_asc",
    "TS_data_vol_asc",
    "TS_exec_time_to_data_vol_ratio_asc",
    "TS_closest_soft_ddl",
    "TS_highest_response_ratio",
]

combinations = list(
    product(lambda_rate_sp, cost_diff_sp, func_num_sp, cpu_core_sp,
            cache_capacity_sp, soft_ddl_param_sp, node_selection_strategy_sp,
            path_selection_strategy_sp, task_sorting_strategy_sp))

start = 0
end = 1

for i in range(start, end):

    # ===指定各个超参数的值===
    lambda_rate = combinations[i][0]  #到达率
    cost_diff = combinations[i][1]  #传输代价
    func_num = combinations[i][2]  #函数集大小
    cpu_core = combinations[i][3]  #CPU核心数
    cache_capacity = combinations[i][4]  #缓存容量
    soft_ddl_param = combinations[i][5]  #软截止期生成参数

    # ===指定各个算法组件的策略===
    node_selection_strategy = combinations[i][6]  #节点选择策略
    path_selection_strategy = combinations[i][7]  #路径选择策略
    task_sorting_strategy = combinations[i][8]  #任务排序策略

    print(
        lambda_rate,
        cost_diff,
        func_num,
        cpu_core,
        cache_capacity,
        soft_ddl_param,
        node_selection_strategy,
        path_selection_strategy,
        task_sorting_strategy,
        AIHO_Plus(lambda_rate, func_num, cost_diff, cpu_core, cache_capacity, soft_ddl_param, node_selection_strategy, path_selection_strategy, task_sorting_strategy),
        sep="|")
