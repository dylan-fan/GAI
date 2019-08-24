import numpy as np

from arch.api import federation
from arch.api.utils import log_utils
from arch.api.utils.log_utils import logDtableInstances
from federatedml.neighbors_sampling.neighbors_sampling import NeighborsSampling
from federatedml.secureprotol import EncryptModeCalculator
from federatedml.util.transfer_variable import NeighborsSamplingTransferVariable
from federatedml.util import consts

LOGGER = log_utils.getLogger()

class NeighborsSamplingGuest(NeighborsSampling):
    def __init__(self, neighbors_sampling_params):
        super(NeighborsSamplingGuest, self).__init__(neighbors_sampling_params)
        self.transfer_variable = NeighborsSamplingTransferVariable()  

    # local neighbors sampling: direct use superclass's

    # distributed neighbors sampling
    def distributed_neighbors_sampling(self, bridge_instances, adj_instances):
        # distributed positive neighbors sampling
        LOGGER.info("bridge_instances count:{}".format(bridge_instances.count()))
        LOGGER.info("Enter distributed neighbors sampling for guest")

        host_distributed_instances_dst = self._distributed_sampling_dst(bridge_instances, 
                                                                   src=consts.HOST,
                                                                   dst=consts.GUEST)
        LOGGER.info("host_distributed_instances_dst count: {}".format(host_distributed_instances_dst.count()))

        guest_distributed_instances_src = self._distributed_sampling_src(bridge_instances,
                                                                        src=consts.GUEST,
                                                                        dst=consts.HOST)
        LOGGER.info("guest_distributed_instances_src count: {}".format(guest_distributed_instances_src.count()))
    

        # distributed negative neighbors sampling
        host_neg_instances_dst = self._distributed_negative_sampling_dst(adj_instances, src=consts.HOST, dst=consts.GUEST)
        LOGGER.info("host_neg_instances_dst count: {}".format(host_neg_instances_dst.count()))
       

        guest_neg_instances_src = self._distributed_negative_sampling_src(guest_distributed_instances_src, 
                                                                          src=consts.GUEST, 
                                                                          dst=consts.HOST)
        LOGGER.info("guest_neg_instances_src count: {}".format(guest_neg_instances_src.count()))

        # union the positive and negative samples
        distributed_instances_dst = host_distributed_instances_dst.union(host_neg_instances_dst)
        LOGGER.info("distributed_instances_dst count: {}".format(distributed_instances_dst.count()))

        distributed_instances_src = guest_distributed_instances_src.union(guest_neg_instances_src)
        LOGGER.info("distributed_instances_src count: {}".format(distributed_instances_src.count()))

        return distributed_instances_dst, distributed_instances_src
        
        

        

        

        





        
        



