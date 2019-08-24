import numpy as np

from arch.api import federation
from arch.api import eggroll
from arch.api.utils import log_utils
from federatedml.network_embedding.base_network_embedding import BaseNetworkEmbeddig
from federatedml.optim.gradient import HeteroNetworkEmbeddingGradient
from federatedml.secureprotol import EncryptModeCalculator
from federatedml.statistic.data_overview import rubbish_clear
from federatedml.util import consts
from federatedml.util.transfer_variable import HeteroNETransferVariable
from federatedml.param import NetworkEmbeddingParam
from federatedml.util.data_io import Instance

LOGGER = log_utils.getLogger()

class HeteroNEHost(BaseNetworkEmbeddig):
    def __init__(self, network_embedding_params: NetworkEmbeddingParam):
        super(HeteroNEHost, self).__init__(network_embedding_params)
        self.transfer_variable = HeteroNETransferVariable()
        self.batch_num = None
        self.batch_index_list = []

    
    @staticmethod
    def load_data(data_instance):
        """
        transform pair data to Instance
        Parameters
        ----------
        data_instance: anchor node
        """
        return Instance(features=data_instance)

    def fit(self, data_instances, node2id):
        """
        Train ne model pf role host
        Parameters
        ----------
        data_instances: Dtable of anchor node, input data
        """
        LOGGER.info("Enter hetero_ne host")
        self.n_node = len(node2id)
        LOGGER.info("Host party has {} nodes".format(self.n_node))

        data_instances = data_instances.mapValues(HeteroNEHost.load_data)
        LOGGER.info("Transform input data to train instance")

        public_key = federation.get(name=self.transfer_variable.paillier_pubkey.name,
                                   tag=self.transfer_variable.generate_transferid(
                                       self.transfer_variable.paillier_pubkey
                                   ),
                                   idx=0)
        LOGGER.info("Get Publick key from arbiter:{}".format(public_key))
        self.encrypt_operator.set_public_key(public_key)

        batch_info = federation.get(name=self.transfer_variable.batch_info.name,
                                    tag=self.transfer_variable.generate_transferid(self.transfer_variable.batch_info),
                                    idx=0)
        LOGGER.info("Get batch_info from guest: {}".format(batch_info))

        self.batch_size = batch_info['batch_size']
        self.batch_num = batch_info['batch_num']
        if self.batch_size < consts.MIN_BATCH_SIZE and self.batch_size != -1:
            raise ValueError(
                            "Batch size get from guest should not less than 10, except -1, batch_size is {}".format(
                            self.batch_size))
        
        self.encrypted_calculator = [EncryptModeCalculator(self.encrypt_operator,
                                                           self.encrypted_mode_calculator_param.mode,
                                                           self.encrypted_mode_calculator_param.re_encrypted_rate)
                                    for _ in range(self.batch_num)]
            
        LOGGER.info("Start initilize model.")
        self.embedding_ = self.initializer.init_model((self.n_node, self.dim), self.init_param_obj)
        
        self.n_iter_ = 0
        index_data_inst_map = {}

        while self.n_iter_ < self.max_iter:
            LOGGER.info("iter: {}".format(self.n_iter_))
            batch_index = 0
            while batch_index < self.batch_num:
                LOGGER.info("batch:{}".format(batch_index))

                # set batch_data
                # in order to avoid communicating in next iteration
                # in next iteration, the sequence of batches is the same
                if len(self.batch_index_list) < self.batch_num:
                    batch_data_index = federation.get(name=self.transfer_variable.batch_data_index.name,
                                                      tag=self.transfer_variable.generate_transferid(
                                                          self.transfer_variable.batch_data_index,
                                                          self.n_iter_,
                                                          batch_index
                                                      ),
                                                      idx=0)
                    LOGGER.info("Get batch_index from Guest")
                    self.batch_index_list.append(batch_index)
                else:
                    batch_data_index = self.batch_index_list[batch_index]
                
                # Get mini-batch train_data
                # in order to avoid joining for next iteration
                if len(index_data_inst_map) < self.batch_num:
                    batch_data_inst = batch_data_index.join(data_instances, lambda g, d: d)
                    index_data_inst_map[batch_index] = batch_data_inst
                else:
                    batch_data_inst = index_data_inst_map[batch_data_index]
                
                LOGGER.info("batch_data_inst size:{}".format(batch_data_inst.count()))

                #self.transform(data_inst)
                
                # compute forward
                host_forward = self.compute_forward(batch_data_inst, self.embedding_, node2id, batch_index)
                federation.remote(host_forward,
                                  name=self.transfer_variable.host_forward_dict.name,
                                  tag=self.transfer_variable.generate_transferid(
                                      self.transfer_variable.host_forward_dict,
                                      self.n_iter_,
                                      batch_index
                                  ),
                                  role=consts.GUEST,
                                  idx=0)
                LOGGER.info("Remote host_forward to guest")

                # Get optimize host gradient and update model
                optim_host_gradient = federation.get(name=self.transfer_variable.host_optim_gradient.name,
                                                     tag=self.transfer_variable.generate_transferid(
                                                         self.transfer_variable.host_optim_gradient,
                                                         self.n_iter_,
                                                         batch_index
                                                     ),
                                                     idx=0)
                LOGGER.info("Get optim_host_gradient from arbiter")
                
                nodeid_join_gradient = batch_data_inst.join(optim_host_gradient, lambda instance, gradient: (node2id[instance.features], gradient))
                LOGGER.info("update_model")
                self.update_model(nodeid_join_gradient)

                # update local model
                #training_info = {"iteration": self.n_iter_, "batch_index": batch_index}
                #self.update_local_model(fore_gradient, batch_data_inst, self.coef_, **training_info)

                batch_index += 1

                rubbish_list = [host_forward]
                rubbish_clear(rubbish_list)

            is_stopped = federation.get(name=self.transfer_variable.is_stopped.name,
                                        tag=self.transfer_variable.generate_transferid(
                                            self.transfer_variable.is_stopped,
                                            self.n_iter_,
                                        ))
            LOGGER.info("Get is_stop flag from arbiter:{}".format(is_stopped))

            self.n_iter_ += 1
            if is_stopped:
                LOGGER.info("Reach max iter {}, train mode finish!".format(self.max_iter))
        
        embedding_table = eggroll.table(name='host', namespace='node_embedding', partition=10)
        id2node = dict(zip(node2id.values(), node2id.keys()))
        for id, embedding in enumerate(self.embedding_):
            embedding_table.put(id2node[id], embedding)
        embedding_table.save_as(name='host', namespace='node_embedding', partition=10)
        LOGGER.info("Reach max iter {}, train model finish!".format(self.max_iter))
        





    