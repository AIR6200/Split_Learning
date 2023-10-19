import argparse, json
import torch, random
from server_handdle import *
from client_handdle import *
from fed_server import *
import models, datasets

from concurrent.futures import ThreadPoolExecutor, as_completed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split Learning & Federated Learning')
    parser.add_argument('-c', '--conf', dest='conf')
    args = parser.parse_args()

    ##读取配置文件信息。
    with open(args.conf, 'r') as f:
        conf = json.load(f)
        print(conf)

    ##分别定义一个服务端对象和多个客户端对象用来模拟分割学习训练场景。
    train_datasets, eval_datasets = datasets.get_dataset("./data/", conf["type"])
    clients = []
    server = Server(conf)
    fedserver = Fed_server(conf)
    #client = Client(conf,train_datasets, eval_datasets)
    for c in range(conf["no_models"]):
        clients.append(Client(conf,train_datasets, eval_datasets, c))

    # 定义客户端选择策略,这里用的是随机选择策略
    candidates = random.sample(clients, conf["k"])


    print("\n\n")



    for e in range(conf["global_epochs"]):
        print("global epochs:", e)
        #初始化模型参数序列，用于后续的全局模型更新和聚合
        server_weight_accumulator = {}
        client_weight_accumulator = {}

        #初始化client端模型序列
        for name, params in fedserver.client_global_model.state_dict().items(): #state_dict()变量存放训练过程中需要学习的权重和偏执系数，state_dict作为python的字典对象将每一层的参数映射成tensor张量
            client_weight_accumulator[name] = torch.zeros_like(params) #生成和括号内变量维度维度一致的全是零的内容

        #初始化server端模型序列
        for name, params in server.global_model.state_dict().items(): #state_dict()变量存放训练过程中需要学习的权重和偏执系数，state_dict作为python的字典对象将每一层的参数映射成tensor张量
            server_weight_accumulator[name] = torch.zeros_like(params) #生成和括号内变量维度维度一致的全是零的内容

        #选择客户端开始训练
        #for c in candidates:
        def local_train(c):
            #客户端计算准确率和损失值
            acc, loss = c.model_eval(fedserver.client_global_model,server)
            print("clientID: %d,acc: %f, loss: %f\n" % (c.client_id, acc, loss))

            diff_client = c.local_train(fedserver.client_global_model)
            for name, params in fedserver.client_global_model.state_dict().items():
                client_weight_accumulator[name].add_(diff_client[name])

            #服务端完成训练后，把最新的server.local_model进行更新聚合
            diff_server = server.local_train_server_model(server.global_model)
            for name, params in server.global_model.state_dict().items():
                server_weight_accumulator[name].add_(diff_server[name])
        
        #引入并发执行
        with ThreadPoolExecutor(max_workers=len(candidates)) as executor:
            executor.map(local_train, candidates)
        
        #更新client端的全局模型（FedServer来做比较合适）
        fedserver.client_model_aggregate(client_weight_accumulator)

        #更新server端的全局模型
        server.model_aggregate(server_weight_accumulator)