import os
import pickle
import models, torch

class Server(object):

    def __init__(self, conf):

        self.conf = conf
        self.local_model = models.get_model(self.conf["server_model_name"])
        self.global_model = models.get_model(self.conf["server_model_name"])

    def local_train(self, client_label,client_model):
        """
        配合客户端完成一轮的SL训练
        model：上一轮训练完成的全局模型
        client_label:客户端传输过来的数据集标签
        client_model：客户端传输过来的客户端模型
        return:客户端端模型的梯度计算结果
        """
        if os.path.exists(self.conf["server_model_pth"]):
            pretrained_model = self.conf["server_model_pth"]
            self.local_model.load_state_dict(torch.load(pretrained_model))

        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'],momentum=self.conf['momentum'])
        self.local_model.train()
        optimizer.zero_grad()

        #接受来自client的模型，继续训练
        output = self.local_model(client_model)
        loss = torch.nn.functional.cross_entropy(output, client_label)
        loss.backward()
        optimizer.step()
        response = pickle.dumps(client_model.grad)

        pth = self.conf["server_model_pth"]
        torch.save(self.local_model.state_dict(), pth)

        return response

    def local_train_server_model(self,model):
        """
        返回server端模型以供后续进行聚合更新
        :param model: 上一轮训练中server端的全局模型
        :return: 经过本轮训练后的server端局部模型
        """
        #这里的model指的是聚合完成之后的全局模型

        if os.path.exists(self.conf["server_model_pth"]):
            pretrained_model = self.conf["server_model_pth"]
            self.local_model.load_state_dict(torch.load(pretrained_model))

        diff = dict()
        for name, data in self.local_model.state_dict().items():
            diff[name] = (data - model.state_dict()[name])
        return diff


    def model_aggregate(self, weight_accumulator):
        for name, data in self.global_model.state_dict().items():
            update_per_layer = weight_accumulator[name]* self.conf["lambda"]
            if data.type() != update_per_layer.type():
                data.add_(update_per_layer.to(torch.int64))
            else:
                data.add_(update_per_layer)


    def model_eval(self,client_label,client_model):

        self.global_model.eval()
        output = self.global_model(client_model)
        loss = torch.nn.functional.cross_entropy(output, client_label, reduction='sum').item()
        pred = output.data.max(1)[1]

        return pickle.dumps(loss),pickle.dumps(pred)
