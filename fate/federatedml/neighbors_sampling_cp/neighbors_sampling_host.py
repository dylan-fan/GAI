import numpy as np

from arch.api import federation
from arch.api.utils import log_utils
from federatedml.neighbors_sampling.neighbors_sampling import NeighborsSampling
from federatedml.secureprotol import EncryptModeCalculator
from federatedml.util.transfer_variable import NeighborsSamplingTransferVariable
from federatedml.util import consts
from arch.api.utils.log_utils import logDtableInstances

LOGGER = log_utils.getLogger()

class NeighborsSamplingHost(NeighborsSampling):
    def __init__(self, neighbors_sampling_params):
        super(NeighborsSamplingHost, self).__init__(neighbors_sampling_params)
        self.transfer_variable = NeighborsSamplingTransferVariable()
    
    # local neighbors sampling: direct use superclass's
    
    # distribute neighbos_sampling
    def distributed_neighbors_sampling(self, bridge_instances, adj_instances):
        # distributed positive neighbors sampling
        LOGGER.info("bridge_instances count:{}".format(bridge_instances.count()))
        LOGGER.info("Enter distributed neighbors sampling for guest")
        
        host_distributed_instances_src = self._distributed_sampling_src(bridge_instances,
                                                                        src=consts.HOST,
                                                                        dst=consts.GUEST)
        LOGGER.info("display top 6 host_distributed_instances_src:")
        logDtableInstances(LOGGER, host_distributed_instances_src, topk=5, isInstance=False)

        guest_distributed_instances_dst = self._distributed_sampling_dst(bridge_instances,
                                                                         src=consts.GUEST,
                                                                         dst=consts.HOST)
        LOGGER.info("display top 6 guest_distributed_instances_dst:")
        logDtableInstances(LOGGER, guest_distributed_instances_dst, topk=5, isInstance=False)

        # distributed negative neighbors sampling
        host_neg_instances_src = self._distributed_negative_sampling_src(host_distributed_instances_src, src=consts.HOST, dst=consts.GUEST)
        
        guest_neg_instances_dst = self._distributed_negative_sampling_dst(adj_instances,
                                                                          src=consts.GUEST,
                                                                          dst=consts.HOST)

        distributed_instances_src = host_distributed_instances_src.union(host_neg_instances_src)
        LOGGER.info("distributed_instances_dst count: {}".format(distributed_instances_src.count()))

        distributed_instances_dst = guest_distributed_instances_dst.union(guest_neg_instances_dst)
        LOGGER.info("distributed_instances_dst count: {}".format(distributed_instances_dst.count()))

        return distributed_instances_dst, distributed_instances_src

       
       
        



