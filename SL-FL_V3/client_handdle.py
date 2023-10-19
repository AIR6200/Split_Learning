import pickle
import os
import models, torch, copy
from torch.autograd import Variable
import numpy as np
import client_communication

class Client(object):

    def __init__(self, conf, train_dataset,eval_dataset, id=-1):
        """
        初始化客户端定义
        conf：加载配置信息
        local_model：加载本地模型
        global_model：加载全局模型
        client_id：客户端编号
        train_dataset：训练数据集
        eval_dataset：测试数据集
        """

        self.conf = conf
        self.local_model = models.get_model(self.conf["client_model_name"])
        self.global_model = models.get_model(self.conf["client_model_name"])
        self.client_id = id

        self.train_dataset = train_dataset

        train_indices=[]

        self.device = torch.device('cuda' if torch.cuda.is_available() & self.conf['GPU'] else 'cpu')

        #TODO:modify the closure
        def get_non_IID_data(dataset,num_users,id):
            """
            Sample non-I.I.D client data from dataset
            :param dataset:
            :param num_users:
            :return:
            """ 
            idxs = np.arange(len(dataset))

            labels = dataset.targets

            # idxs and labels
            idxs_labels = np.vstack((idxs, labels))

            #sort labels
            idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
            idxs = idxs_labels[0,:]
            data_len = int(len(dataset)/ num_users)
            return idxs[id*data_len:(id+1)*data_len]

        #训练集生成:IID分布
        if self.conf["iid"]==True:
            #all_range = list(range(int(len(self.train_dataset)/10)))
            #data_len = int(int(len(self.train_dataset)/ self.conf['no_models'])/10)
            all_range = list(range(len(self.train_dataset)))
            data_len = int(len(self.train_dataset)/ self.conf['no_models'])

            train_indices = all_range[id * data_len: (id + 1) * data_len]
        else:
            train_indices=get_non_IID_data(train_dataset,self.conf['no_models'],id)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=conf["batch_size"],sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices))


        #测试集生成
        #all_range_eval = list(range(len(self.eval_dataset)))
        #data_len_eval = int(len(self.eval_dataset) / self.conf['no_models'])
        #train_indices_eval = all_range_eval[id * data_len_eval: (id + 1) * data_len_eval]
        self.eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.conf["batch_size"], shuffle=True)
        #self.eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.conf["batch_size"], sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices_eval))

    def local_train(self, model):
        """
        实现一次分割学习的训练，包括客户端本地训练及服务端训练

        param
        model: 客户端的全局模型，即上一轮训练后的客户端模型

        return
        client_diff: 本轮训练完成后的本地模型，用于后续的模型聚合更新
        """


        for name, param in model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())

        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'],momentum=self.conf['momentum'])
        self.local_model.train()
        for e in range(self.conf["local_epochs"]):

            for batch_id, batch in enumerate(self.train_loader):
                data, target = batch
                data=data.to(self.device)
                target=target.to(self.device)
                optimizer.zero_grad()
                output = self.local_model(data)

                # 给server来做后面的内容，grads来自server
                grads = client_communication.SL_training_run(self.conf['server_address']["training_address"],target,output)
                grads = pickle.loads(grads.server_model)
                output.backward(grads)
                optimizer.step()

            #print("Epoch %d done." % e)
        client_diff = dict()
        for name, data in self.local_model.state_dict().items():
            client_diff[name] = (data - model.state_dict()[name])
        return client_diff


    def model_eval(self,model,server_config):
        for name, param in model.state_dict().items():
            self.global_model.state_dict()[name].copy_(param.clone())

        self.global_model.eval()

        total_loss = 0.0
        correct = 0
        dataset_size = 0
        for batch_id, batch in enumerate(self.eval_loader):
            data, target = batch
            data=data.to(self.device)
            target=target.to(self.device)
            dataset_size += data.size()[0]

            output = self.global_model(data)

            #给server做后面的内容, loss和pred来自server
            response = client_communication.SL_validation_run(self.conf['server_address']["validation_address"],target,output,server_config)
            loss = pickle.loads(response.loss)
            pred = pickle.loads(response.correct)
            total_loss +=loss
            #correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            correct += pred.eq(target).sum().item()

        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size

        return acc, total_l