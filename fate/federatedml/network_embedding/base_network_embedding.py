#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import numpy as np
from federatedml.optim.gradient import HeteroNetworkEmbeddingGradient, NEGradient
from arch.api.model_manager import manager as model_manager
from arch.api.utils import log_utils
from federatedml.evaluation import Evaluation
from federatedml.optim import EmbeddingInitializer
from federatedml.optim import EmUpdater
from federatedml.param import NetworkEmbeddingParam
from federatedml.secureprotol import PaillierEncrypt, FakeEncrypt
from federatedml.statistic import data_overview
from federatedml.util import NetworkEmbeddingChecker
from federatedml.util import consts                       # constant variables
from federatedml.util import fate_operator, abnormal_detection
from federatedml.statistic.data_overview import rubbish_clear
from federatedml.network_embedding.network_embedding_modelmeta import NetworkEmbeddingModelMeta 

LOGGER = log_utils.getLogger()


class BaseNetworkEmbeddig(object):
    def __init__(self, network_embedding_params: NetworkEmbeddingParam):
        self.param = network_embedding_params
        # set params
        NetworkEmbeddingChecker.check_param(network_embedding_params)
        self.dim = network_embedding_params.dim
        self.n_node = network_embedding_params.n_node
        self.init_param_obj = network_embedding_params.init_param
        self.learning_rate = network_embedding_params.learning_rate
        self.encrypted_mode_calculator_param = network_embedding_params.encrypted_mode_calculator_param
        self.encrypted_calculator = None

        self.local_gradient_operator = NEGradient()
        self.updater = EmUpdater()

        self.eps = network_embedding_params.eps
        self.batch_size = network_embedding_params.batch_size
        self.max_iter = network_embedding_params.max_iter
        

        if network_embedding_params.encrypt_param.method == consts.PAILLIER:
            self.encrypt_operator = PaillierEncrypt()
        else:
            self.encrypt_operator = FakeEncrypt()

        # attribute:
        self.n_iter_ = 0
        self.embedding_ = None

        self.gradient_operator = None
        self.initializer = EmbeddingInitializer()
        self.transfer_variable = None
        self.model_meta = NetworkEmbeddingModelMeta()
        self.loss_history = []
        self.is_converged = False
        self.class_name = self.__class__.__name__

    def set_flowid(self, flowid=0):
        if self.transfer_variable is not None:
            self.transfer_variable.set_flowid(flowid)
            LOGGER.debug("set flowid:" + str(flowid))

    def compute_logits(self, data_instances, embedding):
        """
        compute e_i.dot(e_j)
        """
        pass


    def compute_embedding(self, data_instance, embedding_, node2id: dict):
        """
        embedding_lookup depends on the node id
        """
        e = data_instance.mapValues(lambda v: embedding_[node2id[v.features]])
        return e
    
    def compute_local_embedding(self, data_instance, embedding_, node2id):
        E_Y = data_instance.mapValues(lambda v:(embedding_[node2id[v[0]]], embedding_[node2id[v[1]]], v[2]))
        return E_Y
        
    
    def compute_forward(self, data_instance, embedding_, node2id, batch_index=-1):
        """
            Compute e_guest, where e_guest is the node embedding which stays on guest
        """
        e = self.compute_embedding(data_instance, embedding_, node2id)
        en_e = self.encrypted_calculator[batch_index].encrypt(e)
        
        forward = en_e.join(e, lambda en_e, e: (en_e, e))

        rubbish_list = [e, en_e]
        rubbish_clear(rubbish_list)
        return forward

    def update_model(self, nodeid_join_gradient):
        """
        update node embedding according to the gradient
        Paramters
        ---------
        node_join_gradient: DTable, key=sample_id, value=(node_id, gradient) 
        """
        self.embedding_ = self.updater.update_embedding(self.embedding_, nodeid_join_gradient)

    def update_common_nodes(self, common_embedding, common_nodes, node2id):
        for node in common_nodes:
            self.embedding_[node2id[node]] = common_embedding.get(node)


    def fit(self, data_instance, node2id):
        pass

    def predict(self, data_instance, predict_param):
        pass

    def _save_meta(self, name, namespace):
        raise NotImplementedError("save_meta has not been implemented yet!")
    
    def save_model(self, name, namespace):
        pass

    def load_model(self, name, namespace):
        pass

    def _abnormal_detection(self, data_instances):
        """
        Make sure input data_instances is valid.
        """
        abnormal_detection.empty_table_detection(data_instances)
        abnormal_detection.empty_feature_detection(data_instances)

    def show_meta(self):
        pass

    def show_model(self):
        model_dict = {
            'iters': self.n_iter_,
            'loss_history': self.loss_history,
            'is_converged': self.is_converged,
            'embedding': self.embedding_,
        }
        LOGGER.info("Showing model information:")
        for k, v in model_dict.items():
            LOGGER.info("{} is {}".format(k, v))
    
    def update_local_model(self, fore_gradient, data_inst, embedding, **training_info):
        """
        For inductive learning

        Parameters:
        ___________
        :param fore_gradient: a table holding fore gradient
        :param data_inst: a table holding instances of raw input of guest side
        :param embedding: node embedding
        :param training_info: a dictionary holding training information
        """
        pass

    def transform(self, data_inst):
        """
        For inductive learning, transform node attribute into embedding through neural networks

        Parameters:
        ___________
        :param data_inst: a table holding instances of raw input of guest side
        :return: a table holding instances with transformed features
        """
        return data_inst
