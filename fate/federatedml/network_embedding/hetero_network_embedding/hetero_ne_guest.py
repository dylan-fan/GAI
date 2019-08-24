import numpy as np
import math

from arch.api import federation
from arch.api import eggroll
from arch.api.utils import log_utils
from federatedml.network_embedding.base_network_embedding import BaseNetworkEmbeddig
from federatedml.optim.gradient import HeteroNetworkEmbeddingGradient, NEGradient
from federatedml.secureprotol import EncryptModeCalculator
from federatedml.statistic.data_overview import rubbish_clear
from federatedml.util import consts
from federatedml.util.transfer_variable import HeteroNETransferVariable
from federatedml.util import fate_operator
from federatedml.util.data_io import Instance
from federatedml.model_selection import MiniBatch
from federatedml.optim import Optimizer

LOGGER = log_utils.getLogger()

class HeteroNEGuest(BaseNetworkEmbeddig):
    def __init__(self, network_embedding_params):
        super(HeteroNEGuest, self).__init__(network_embedding_params)
        self.transfer_variable = HeteroNETransferVariable()
        self.data_batch_count = []
        
        self.encrypted_calculator = None

        self.guest_forward = None

        ######
        self.local_optimizer = Optimizer(network_embedding_params.learning_rate, 
                                     network_embedding_params.optimizer)
        ######

    def aggregate_forward(self, host_forward):
        """
        Compute e_guest.dot(e_host)
        Paramters
        ---------
        host_forward: DTable. key, en_e(host)
        """
        aggregate_forward_res = self.guest_forward.join(host_forward, 
                                                        lambda e1, e2: 
                                                        (fate_operator.dot(e1[1], e2[1]), 
                                                         math.pow(fate_operator.dot(e1[1], e2[1]), 2)
                                                        )
                                                       )
        return aggregate_forward_res

    @staticmethod
    def load_data(data_instance):
        """
        transform pair data to Instance
        Parameters
        ----------
        data_instance: tuple (node, label)
        """
        return Instance(features=data_instance[0], label=data_instance[1])

    def fit(self, data_instances, node2id, local_instances=None, common_nodes=None):
        """
        Train node embedding for role guest
        Parameters
        ----------
        data_instances: DTable of target node and label, input data
        node2id: a dict which can map node name to id
        """
        LOGGER.info("samples number:{}".format(data_instances.count()))
        LOGGER.info("Enter network embedding procedure:")
        self.n_node = len(node2id)
        LOGGER.info("Bank A has {} nodes".format(self.n_node))

        data_instances = data_instances.mapValues(HeteroNEGuest.load_data)
        LOGGER.info("Transform input data to train instance")

        public_key = federation.get(name=self.transfer_variable.paillier_pubkey.name,
                                    tag=self.transfer_variable.generate_transferid(
                                        self.transfer_variable.paillier_pubkey
                                    ),
                                    idx=0)
        LOGGER.info("Get public_key from arbiter:{}".format(public_key))
        self.encrypt_operator.set_public_key(public_key)

        # hetero network embedding
        LOGGER.info("Generate mini-batch from input data")
        mini_batch_obj = MiniBatch(data_instances, batch_size=self.batch_size)
        batch_num = mini_batch_obj.batch_nums

        LOGGER.info("samples number:{}".format(data_instances.count()))
        if self.batch_size == -1:
            LOGGER.info("batch size is -1, set it to the number of data in data_instances")
            self.batch_size = data_instances.count()


        ##############
        # horizontal federated learning
        LOGGER.info("Generate mini-batch for local instances in guest")
        mini_batch_obj_local = MiniBatch(local_instances, batch_size=self.batch_size)
        local_batch_num = mini_batch_obj_local.batch_nums
        common_node_instances = eggroll.parallelize(((node, node) for node in common_nodes), include_key=True, name='common_nodes')
        ##############

        batch_info = {'batch_size': self.batch_size, "batch_num": batch_num}
        LOGGER.info("batch_info:{}".format(batch_info))
        federation.remote(batch_info,
                          name=self.transfer_variable.batch_info.name,
                          tag=self.transfer_variable.generate_transferid(self.transfer_variable.batch_info),
                          role=consts.HOST,
                          idx=0)
        LOGGER.info("Remote batch_info to Host")

        federation.remote(batch_info,
                          name=self.transfer_variable.batch_info.name,
                          tag=self.transfer_variable.generate_transferid(self.transfer_variable.batch_info),
                          role=consts.ARBITER,
                          idx=0)
        LOGGER.info("Remote batch_info to Arbiter")
        
        self.encrypted_calculator = [EncryptModeCalculator(self.encrypt_operator,
                                                       self.encrypted_mode_calculator_param.mode,
                                                       self.encrypted_mode_calculator_param.re_encrypted_rate)
                                                       for _ in range(batch_num)]
        
        LOGGER.info("Start initialize model.")
        self.embedding_ = self.initializer.init_model((self.n_node, self.dim), self.init_param_obj)
        LOGGER.info("Embedding shape={}".format(self.embedding_.shape))

        is_send_all_batch_index=False
        self.n_iter_ = 0
        index_data_inst_map = {}

        while self.n_iter_ < self.max_iter:
            LOGGER.info("iter:{}".format(self.n_iter_))

            #################
            local_batch_data_generator = mini_batch_obj_local.mini_batch_data_generator()
            total_loss = 0
            local_batch_num = 0
            LOGGER.info("Enter the horizontally federated learning procedure:")
            for local_batch_data in local_batch_data_generator:
                n = local_batch_data.count()
                #LOGGER.info("Local batch data count:{}".format(n))
                E_Y = self.compute_local_embedding(local_batch_data, self.embedding_, node2id)
                local_grads_e1, local_grads_e2, local_loss = self.local_gradient_operator.compute(E_Y, 'E_1')
                local_grads_e1 = local_grads_e1.mapValues(lambda g: self.local_optimizer.apply_gradients(g/n))
                local_grads_e2 = local_grads_e2.mapValues(lambda g: self.local_optimizer.apply_gradients(g/n))
                e1id_join_grads = local_batch_data.join(local_grads_e1, lambda v, g: (node2id[v[0]], g))
                e2id_join_grads = local_batch_data.join(local_grads_e2, lambda v, g: (node2id[v[1]], g))
                self.update_model(e1id_join_grads)
                self.update_model(e2id_join_grads)

                local_loss = local_loss / n
                local_batch_num += 1
                total_loss += local_loss
                #LOGGER.info("gradient count:{}".format(e1id_join_grads.count()))

            guest_common_embedding = common_node_instances.mapValues(lambda node: self.embedding_[node2id[node]])
            federation.remote(guest_common_embedding,
                                name=self.transfer_variable.guest_common_embedding.name,
                                tag=self.transfer_variable.generate_transferid(
                                    self.transfer_variable.guest_common_embedding,
                                    self.n_iter_,
                                    0
                                ),
                                role=consts.ARBITER,
                                idx=0)
            LOGGER.info("Remote the embedding of common node to arbiter!")

            common_embedding = federation.get(name=self.transfer_variable.common_embedding.name,
                                              tag=self.transfer_variable.generate_transferid(
                                                  self.transfer_variable.common_embedding,
                                                  self.n_iter_,
                                                  0
                                              ),
                                              idx=0)
            LOGGER.info("Get the aggregated embedding of common node from arbiter!")
            
            self.update_common_nodes(common_embedding, common_nodes, node2id)
            
            total_loss /= local_batch_num
            LOGGER.info("Iter {}, horizontally feaderated learning loss: {}".format(self.n_iter_, total_loss))


            #################

            # verticallly feaderated learning
            # each iter will get the same batch_data_generator
            LOGGER.info("Enter the vertically federated learning:")
            batch_data_generator = mini_batch_obj.mini_batch_data_generator(result='index')

            batch_index = 0
            for batch_data_index in batch_data_generator:
                LOGGER.info("batch:{}".format(batch_index))

                # only need to send one times
                if not is_send_all_batch_index:
                    LOGGER.info("remote mini-batch index to Host")
                    federation.remote(batch_data_index, 
                                      name=self.transfer_variable.batch_data_index.name,
                                      tag=self.transfer_variable.generate_transferid(
                                          self.transfer_variable.batch_data_index,
                                          self.n_iter_,
                                          batch_index
                                      ),
                                      role=consts.HOST,
                                      idx=0)
                    if batch_index >= mini_batch_obj.batch_nums - 1:
                        is_send_all_batch_index = True
                
                # in order to avoid joining in next iteration
                # Get mini-batch train data
                if len(index_data_inst_map) < batch_num:
                    batch_data_inst = data_instances.join(batch_data_index, lambda data_inst, index: data_inst)
                    index_data_inst_map[batch_index] = batch_data_inst
                else:
                    batch_data_inst = index_data_inst_map[batch_index]
                
                # For inductive learning: transform node attributes to node embedding
                # self.transform(batch_data_inst)
                self.guest_forward = self.compute_forward(batch_data_inst, self.embedding_, node2id, batch_index)

                host_forward = federation.get(name=self.transfer_variable.host_forward_dict.name,
                                              tag=self.transfer_variable.generate_transferid(
                                                  self.transfer_variable.host_forward_dict,
                                                  self.n_iter_,
                                                  batch_index
                                              ),
                                              idx=0)
                LOGGER.info("Get host_forward from host")
                aggregate_forward_res = self.aggregate_forward(host_forward)
                en_aggregate_ee = aggregate_forward_res.mapValues(lambda v: v[0])
                en_aggregate_ee_square = aggregate_forward_res.mapValues(lambda v: v[1])

                # compute [[d]]
                if self.gradient_operator is None:
                    self.gradient_operator = HeteroNetworkEmbeddingGradient(self.encrypt_operator)
                fore_gradient = self.gradient_operator.compute_fore_gradient(batch_data_inst, en_aggregate_ee)

                host_gradient = self.gradient_operator.compute_gradient(self.guest_forward.mapValues(lambda v: Instance(features=v[1])), fore_gradient)
                federation.remote(host_gradient,
                                  name=self.transfer_variable.host_gradient.name,
                                  tag=self.transfer_variable.generate_transferid(
                                      self.transfer_variable.host_gradient,
                                      self.n_iter_,
                                      batch_index
                                  ),
                                  role=consts.ARBITER,
                                  idx=0)
                LOGGER.info("Remote host_gradient to arbiter")
                
                composed_data_inst = host_forward.join(batch_data_inst, lambda hf, d: Instance(features=hf[1], label=d.label))
                guest_gradient, loss = self.gradient_operator.compute_gradient_and_loss(composed_data_inst,
                                                                                        fore_gradient,
                                                                                        en_aggregate_ee,
                                                                                        en_aggregate_ee_square)
                federation.remote(guest_gradient,
                                  name=self.transfer_variable.guest_gradient.name,
                                  tag=self.transfer_variable.generate_transferid(
                                      self.transfer_variable.guest_gradient,
                                      self.n_iter_,
                                      batch_index
                                  ),
                                  role=consts.ARBITER,
                                  idx=0)
                LOGGER.info("Remote guest_gradient to arbiter")

                optim_guest_gradient = federation.get(name=self.transfer_variable.guest_optim_gradient.name,
                                                      tag=self.transfer_variable.generate_transferid(
                                                          self.transfer_variable.guest_optim_gradient,
                                                          self.n_iter_,
                                                          batch_index
                                                      ),
                                                      idx=0)
                LOGGER.info("Get optim_guest_gradient from arbiter")

                # update node embedding
                LOGGER.info("Update node embedding")
                nodeid_join_gradient = batch_data_inst.join(optim_guest_gradient, lambda instance, gradient: (node2id[instance.features], gradient))
                self.update_model(nodeid_join_gradient)

                # update local model that transform attribute to node embedding
                training_info = {'iteration': self.n_iter_, 'batch_index': batch_index}
                self.update_local_model(fore_gradient, batch_data_inst, self.embedding_, **training_info)

                # loss need to be encrypted !!!!!!

                federation.remote(loss, 
                                  name=self.transfer_variable.loss.name,
                                  tag=self.transfer_variable.generate_transferid(
                                      self.transfer_variable.loss,
                                      self.n_iter_,
                                      batch_index
                                      ),
                                  role=consts.ARBITER,
                                  idx=0)
                LOGGER.info("Remote loss to arbiter")

                # is converge of loss in arbiter
                batch_index += 1

                # remove temporary resource
                rubbish_list = [host_forward,
                                aggregate_forward_res,
                                en_aggregate_ee,
                                en_aggregate_ee_square,
                                fore_gradient,
                                self.guest_forward]
                rubbish_clear(rubbish_list)
            
            ##########
            guest_common_embedding = common_node_instances.mapValues(lambda node: self.embedding_[node2id[node]])
            federation.remote(guest_common_embedding,
                                name=self.transfer_variable.guest_common_embedding.name,
                                tag=self.transfer_variable.generate_transferid(
                                    self.transfer_variable.guest_common_embedding,
                                    self.n_iter_,
                                    1
                                ),
                                role=consts.ARBITER,
                                idx=0)
                
            common_embedding = federation.get(name=self.transfer_variable.common_embedding.name,
                                              tag=self.transfer_variable.generate_transferid(
                                                  self.transfer_variable.common_embedding,
                                                  self.n_iter_,
                                                  1
                                              ),
                                              idx=0)
            
            self.update_common_nodes(common_embedding, common_nodes, node2id)
            ##########

            is_stopped = federation.get(name=self.transfer_variable.is_stopped.name,
                                        tag=self.transfer_variable.generate_transferid(
                                            self.transfer_variable.is_stopped,
                                            self.n_iter_
                                        ),
                                        idx=0)
            
            LOGGER.info("Get is_stop flag from arbiter:{}".format(is_stopped))

            self.n_iter_ += 1
            if is_stopped:
                LOGGER.info("Get stop signal from arbiter, model is converged, iter:{}".format(self.n_iter_))
                break
                
        embedding_table = eggroll.table(name='guest', namespace='node_embedding', partition=10)
        id2node = dict(zip(node2id.values(), node2id.keys()))
        for id, embedding in enumerate(self.embedding_):
            embedding_table.put(id2node[id], embedding)
        embedding_table.save_as(name='guest', namespace='node_embedding', partition=10)
        LOGGER.info("Reach max iter {}, train model finish!".format(self.max_iter))
        


    
