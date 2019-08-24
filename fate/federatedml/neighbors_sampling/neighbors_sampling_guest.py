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

        host_distributed_instances_target = self._distributed_sampling_target(bridge_instances, 
                                                                   anchor=consts.HOST,
                                                                   target=consts.GUEST)
        LOGGER.info("host_distributed_instances_target count: {}".format(host_distributed_instances_target.count()))

        guest_distributed_instances_anchor = self._distributed_sampling_anchor(bridge_instances,
                                                                        anchor=consts.GUEST,
                                                                        target=consts.HOST)
        LOGGER.info("guest_distributed_instances_anchor count: {}".format(guest_distributed_instances_anchor.count()))
    

        # distributed negative neighbors sampling
        host_neg_instances_target = self._distributed_negative_sampling_target(adj_instances, anchor=consts.HOST, target=consts.GUEST)
        LOGGER.info("host_neg_instances_target count: {}".format(host_neg_instances_target.count()))
       

        guest_neg_instances_anchor = self._distributed_negative_sampling_anchor(guest_distributed_instances_anchor, 
                                                                          anchor=consts.GUEST, 
                                                                          target=consts.HOST)
        LOGGER.info("guest_neg_instances_anchor count: {}".format(guest_neg_instances_anchor.count()))

        # union the positive and negative samples
        distributed_instances_target = host_distributed_instances_target.union(host_neg_instances_target)
        LOGGER.info("distributed_instances_target count: {}".format(distributed_instances_target.count()))

        distributed_instances_anchor = guest_distributed_instances_anchor.union(guest_neg_instances_anchor)
        LOGGER.info("distributed_instances_anchor count: {}".format(distributed_instances_anchor.count()))

        logDtableInstances(LOGGER, host_distributed_instances_target, topk=3, isInstance=False)
        logDtableInstances(LOGGER, host_neg_instances_target, topk=3, isInstance=False)

        return distributed_instances_target, distributed_instances_anchor
        
        

        

        

        





        
        



