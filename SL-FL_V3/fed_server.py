import os
import pickle
import models, torch

class Fed_server(object):

    def __init__(self, conf):

        self.conf = conf
        self.client_global_model = models.get_model(self.conf["client_model_name"])
        #self.server_global_model = models.get_model(self.conf["server_model_name"])

    def client_model_aggregate(self, weight_accumulator):
        for name, data in self.client_global_model.state_dict().items():
            update_per_layer = weight_accumulator[name]* self.conf["lambda"]
            if data.type() != update_per_layer.type():
                data.add_(update_per_layer.to(torch.int64))
            else:
                data.add_(update_per_layer)
        #pth = self.conf["server_model_pth"]
        #torch.save(self.global_model.state_dict(), pth)
        #print("server hash", hash(self.global_model))

#    def server_model_aggregate(self, weight_accumulator):
#        for name, data in self.server_global_model.state_dict().items():
#            update_per_layer = weight_accumulator[name]* self.conf["lambda"]
#            if data.type() != update_per_layer.type():
#                data.add_(update_per_layer.to(torch.int64))
#            else:
#                data.add_(update_per_layer)