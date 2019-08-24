import numpy as np

from arch.api import federation
from arch.api.utils import log_utils
from federatedml.network_embedding.base_network_embedding import BaseNetworkEmbeddig
from federatedml.optim import Optimizer
from federatedml.optim.convergence import DiffConverge
from federatedml.optim.federated_aggregator import HeteroFederatedAggregator
from federatedml.util import HeteroNETransferVariable
from federatedml.util import consts
from federatedml.param import NetworkEmbeddingParam

LOGGER = log_utils.getLogger()

class HeteroNEArbiter(BaseNetworkEmbeddig):
    def __init__(self, network_embedding_params: NetworkEmbeddingParam):
        super(HeteroNEArbiter, self).__init__(network_embedding_params)
        self.converge_func = DiffConverge(network_embedding_params.eps)

        # attribute
        self.pre_loss = None
        self.batch_num = None
        self.transfer_variable = HeteroNETransferVariable()
        self.optimizer = Optimizer(network_embedding_params.learning_rate, 
                                   network_embedding_params.optimizer)
        
        self.key_length = network_embedding_params.encrypt_param.key_length

    def perform_subtasks(self, **training_info):
        """
        performs any tasks that the arbiter is responsible for.

        This 'perform_subtasks' function serves as a handler on conducting any task that the arbiter is responsible
        for. For example, for the 'perform_subtasks' function of 'HeteroDNNLRArbiter' class located in
        'hetero_dnn_lr_arbiter.py', it performs some works related to updating/training local neural networks of guest
        or host.

        For this particular class (i.e., 'HeteroLRArbiter') that serves as a base arbiter class for neural-networks-based
        hetero-logistic-regression model, the 'perform_subtasks' function will do nothing. In other words, no subtask is
        performed by this arbiter.

        :param training_info: a dictionary holding training information
        """
        pass

    def fit(self, data_instances=None, node2id=None, local_instances=None, common_nodes=None):
        """
        Train network embedding of role arbiter
        Parameters
        ----------
        data_instances: DTable of Instance, input data
        """

        LOGGER.info("Enter hetero_ne_arbiter fit")
        
        # data_instance handele ?

        # Generate encrypt keys
        self.encrypt_operator.generate_key(self.key_length)
        public_key = self.encrypt_operator.get_public_key()
        LOGGER.info("public_key: {}".format(public_key))

        # remote public key to host and guest
        federation.remote(public_key,
                          name=self.transfer_variable.paillier_pubkey.name,
                          tag=self.transfer_variable.generate_transferid(self.transfer_variable.paillier_pubkey),
                          role=consts.HOST,
                          idx=0)
        LOGGER.info("remote publick_key to host")

        federation.remote(public_key,
                          name=self.transfer_variable.paillier_pubkey.name,
                          tag=self.transfer_variable.generate_transferid(self.transfer_variable.paillier_pubkey),
                          role=consts.GUEST,
                          idx=0)
        LOGGER.info("remote public_key to guest")

        batch_info = federation.get(name=self.transfer_variable.batch_info.name,
                                    tag=self.transfer_variable.generate_transferid(self.transfer_variable.batch_info),
                                    idx=0)
        LOGGER.info("Get batch_info from guest: {}".format(batch_info))
        self.batch_num = batch_info['batch_num']
        
        is_stop = False
        self.n_iter_ = 0

        while self.n_iter_ < self.max_iter:
            LOGGER.info("iter: {}".format(self.n_iter_))

             #######
            # Horizontally learning
            host_common_embedding = federation.get(name=self.transfer_variable.host_common_embedding.name,
                                                   tag=self.transfer_variable.generate_transferid(
                                                       self.transfer_variable.host_common_embedding,
                                                       self.n_iter_,
                                                       0
                                                   ),
                                                   idx=0)
            guest_common_embedding = federation.get(name=self.transfer_variable.guest_common_embedding.name,
                                                    tag=self.transfer_variable.generate_transferid(
                                                        self.transfer_variable.guest_common_embedding,
                                                        self.n_iter_,
                                                        0
                                                    ),
                                                    idx=0)

            common_embedding =  host_common_embedding.join(guest_common_embedding, lambda host, guest: (host + guest) / 2)

            federation.remote(common_embedding,
                              name=self.transfer_variable.common_embedding.name,
                              tag=self.transfer_variable.generate_transferid(
                                  self.transfer_variable.common_embedding,
                                  self.n_iter_,
                                  0
                              ),
                              role=consts.HOST,
                              idx=0)
            
            federation.remote(common_embedding,
                              name=self.transfer_variable.common_embedding.name,
                              tag=self.transfer_variable.generate_transferid(
                                  self.transfer_variable.common_embedding,
                                  self.n_iter_,
                                  0
                              ),
                              role=consts.GUEST,
                              idx=0)

            LOGGER.info("Iter {}, horizontally learning finish".format(self.n_iter_))

            #######

            batch_index = 0
            iter_loss = 0

            while batch_index < self.batch_num:
                LOGGER.info("batch: {}".format(batch_index))

                # host_gradient shape = (batch_size, dim)
                host_gradient = federation.get(name=self.transfer_variable.host_gradient.name,
                                             tag=self.transfer_variable.generate_transferid( 
                                                 self.transfer_variable.host_gradient, self.n_iter_, batch_index),
                                             idx=0
                                             )
                LOGGER.info("Get host_gradient from GUEST")

                # guest_gradient DTable key = sample_id, value=gradient(shape=(batch_size, dim))
                guest_gradient = federation.get(name=self.transfer_variable.guest_gradient.name,
                                                tag=self.transfer_variable.generate_transferid(
                                                    self.transfer_variable.guest_gradient, self.n_iter_, batch_index),
                                                idx=0)
                LOGGER.info("Get guest_gradient from GUEST")

                
                #host_gradient, guest_gradient = np.array(host_gradient), np.array(guest_gradient)
                #gradient = np.vstack((host_gradient, guest_gradient))
                
                # if the gradients are encrypted, remember to decrypt
                # for i in range(gradient.shape[0]):
                #     for j in range(gradient.shape[1]):
                #         gradient[i, j] = self.encrypt_operator.decrypt(gradient[i, j])
                
                #optim_gradient = self.optimizer.apply_gradients(gradient)

                #host_optim_gradient = optim_gradient[: host_gradient.shape[0], :]
                #guest_optim_gradient = optim_gradient[host_gradient.shape[0]:, :]
                host_optim_gradient = host_gradient.mapValues(self.optimizer.apply_gradients)
                guest_optim_gradient = guest_gradient.mapValues(self.optimizer.apply_gradients)

                LOGGER.info("host gradients number: {}".format(host_optim_gradient.count()))
                LOGGER.info("guest gradients number: {}".format(guest_optim_gradient.count()))  

                federation.remote(host_optim_gradient,
                                  name=self.transfer_variable.host_optim_gradient.name,
                                  tag=self.transfer_variable.generate_transferid(
                                      self.transfer_variable.host_optim_gradient, 
                                      self.n_iter_, 
                                      batch_index
                                  ),
                                  role=consts.HOST,
                                  idx=0)
                LOGGER.info("Remote host_optim_gradient to Host")

                federation.remote(guest_optim_gradient,
                                  name=self.transfer_variable.guest_optim_gradient.name,
                                  tag=self.transfer_variable.generate_transferid(
                                      self.transfer_variable.guest_optim_gradient,
                                      self.n_iter_,
                                      batch_index 
                                  ),
                                  role=consts.GUEST,
                                  idx=0)
                LOGGER.info("Remote guest_optim_gradient to Guest")
                
                training_info = {"iteration": self.n_iter_, "batch_index": batch_index}
                self.perform_subtasks(**training_info)

                loss = federation.get(name=self.transfer_variable.loss.name,
                                      tag=self.transfer_variable.generate_transferid(
                                          self.transfer_variable.loss,
                                          self.n_iter_,
                                          batch_index
                                      ),
                                      idx=0)

                #de_loss = self.encrypt_operator.decrypt(loss)
                de_loss = loss
                LOGGER.info("Get loss from guest: {}".format(de_loss))
                iter_loss += de_loss


                batch_index += 1
            
            loss = iter_loss / self.batch_num
            LOGGER.info("iter loss: {}".format(loss))

            ########
            host_common_embedding = federation.get(name=self.transfer_variable.host_common_embedding.name,
                                                   tag=self.transfer_variable.generate_transferid(
                                                       self.transfer_variable.host_common_embedding,
                                                       self.n_iter_,
                                                       1
                                                   ),
                                                   idx=0)
            guest_common_embedding = federation.get(name=self.transfer_variable.guest_common_embedding.name,
                                                    tag=self.transfer_variable.generate_transferid(
                                                        self.transfer_variable.guest_common_embedding,
                                                        self.n_iter_,
                                                        1
                                                    ),
                                                    idx=0)

            common_embedding =  host_common_embedding.join(guest_common_embedding, lambda host, guest: (host + guest) / 2)

            federation.remote(common_embedding,
                              name=self.transfer_variable.common_embedding.name,
                              tag=self.transfer_variable.generate_transferid(
                                  self.transfer_variable.common_embedding,
                                  self.n_iter_,
                                  1
                              ),
                              role=consts.HOST,
                              idx=0)
            
            federation.remote(common_embedding,
                              name=self.transfer_variable.common_embedding.name,
                              tag=self.transfer_variable.generate_transferid(
                                  self.transfer_variable.common_embedding,
                                  self.n_iter_,
                                  1
                              ),
                              role=consts.GUEST,
                              idx=0)
            ########


            if self.converge_func.is_converge(loss):
                is_stop = True
            
            federation.remote(is_stop,
                              name=self.transfer_variable.is_stopped.name,
                              tag=self.transfer_variable.generate_transferid(
                                  self.transfer_variable.is_stopped,
                                  self.n_iter_
                              ),
                              role=consts.HOST,
                              idx=0)
            LOGGER.info("Remote is_stop to host: {}".format(is_stop))

            federation.remote(is_stop,
                              name=self.transfer_variable.is_stopped.name,
                              tag=self.transfer_variable.generate_transferid(
                                  self.transfer_variable.is_stopped,
                                  self.n_iter_
                              ),
                              role=consts.GUEST,
                              idx=0)
            LOGGER.info("Remote is_stop to guest: {}".format(is_stop))

            self.n_iter_ += 1
            if is_stop:
                LOGGER.info("Model is converged, iter: {}".format(self.n_iter_))
                break
        LOGGER.info("Reach max iter {} or convergence, train model finish!".format(self.max_iter))

