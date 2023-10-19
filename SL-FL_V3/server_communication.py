import json
import os
import pickle
import time

import torch

from proto import communication_pb2 as pb2
from proto import communication_pb2_grpc as pb2_grpc
import grpc
from concurrent import futures
import server_handdle

with open("./utils/conf.json", 'r') as f:
    conf = json.load(f)
server = server_handdle.Server(conf)

class SLFL_Training(pb2_grpc.TrainingServicer):
    def SL(self, request, context):
        label = pickle.loads(request.label)
        client_model = pickle.loads(request.client_model)
        response = server_handdle.Server.local_train(server,label,client_model)
        return pb2.SL_Server(server_model=response)

class SLFL_Validation(pb2_grpc.ValidationServicer):
    def val(self, request, context):
        label = pickle.loads(request.label_eval)
        client_model = pickle.loads(request.client_model)
        server_config = pickle.loads(request.server_config_eval)
        loss, pred = server_handdle.Server.model_eval(server_config,label,client_model)
        return pb2.ValidationResponse(loss=loss, correct=pred)

def run():
    grpc_server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=64),
        options=[
        ('grpc.max_send_message_length', 100 * 1024 * 1024),
        ('grpc.max_receive_message_length', 100 * 1024 * 1024)]
    )
    # 注册服务
    pb2_grpc.add_TrainingServicer_to_server(SLFL_Training(),grpc_server)
    pb2_grpc.add_ValidationServicer_to_server(SLFL_Validation(),grpc_server)
    # 绑定端口
    grpc_server.add_insecure_port(server.conf['server_address']["training_address"])
    grpc_server.add_insecure_port(server.conf['server_address']["validation_address"])
    print("training server will start in %s" % server.conf['server_address']["training_address"])
    print("validation server will start in %s" % server.conf['server_address']["validation_address"])
    # 启动服务
    grpc_server.start()
    try:
        while 1:
            time.sleep(3600)
    except KeyboardInterrupt:
        grpc_server.stop(0)

if __name__ == '__main__':
    run()

