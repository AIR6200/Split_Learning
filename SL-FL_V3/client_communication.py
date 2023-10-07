import pickle
from proto import communication_pb2 as pb2
from proto import communication_pb2_grpc as pb2_grpc
import grpc

def SL_training_run(serverIP,labels,split_layer_tensor):
    """
    实现与服务端的GRPC通信，完成一轮完整训练

    param
    serverIP: 服务端的IP地址,
    labels: 客户端传递给服务端的数据集的标签,
    split_layer_tensor: 客户端利用本地模型训练后的模型，交给服务端做后续训练

    return
    grads: 服务端完成训练后的梯度信息，用于客户端更新本地模型的参数
    """
    #定义一个频道，地址和服务端一致
    conn = grpc.insecure_channel(serverIP)
    #定义客户端
    client = pb2_grpc.TrainingStub(channel=conn)
    #与server的调用
    grads = client.SL(pb2.SL_Client(
        label=pickle.dumps(labels),
        client_model=pickle.dumps(split_layer_tensor)
    ))
    return grads

def SL_validation_run(serverIP,labels,split_layer_tensor,server_config):
    """
    实现与服务端的GRPC通信，完成准确率、损失值的计算

    param
    serverIP: 服务端的IP地址,
    labels: 客户端传递给服务端的数据集的标签,
    split_layer_tensor: 客户端利用本地模型训练后的模型，交给服务端做后续训练,
    server_config: 服务端的配置信息（其实不要也可以）

    return
    response: 服务端计算后的准确率和损失值
    """

    #定义一个频道，地址和服务端一致
    conn = grpc.insecure_channel(serverIP)
    #定义客户端
    client = pb2_grpc.ValidationStub(channel=conn)
    response = client.val(pb2.SL_Client_Eval(
        label_eval=pickle.dumps(labels),
        client_model=pickle.dumps(split_layer_tensor),
        server_config_eval = pickle.dumps(server_config)
    ))

    return response


