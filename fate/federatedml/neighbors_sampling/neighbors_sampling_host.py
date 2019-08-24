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
        
        host_distributed_instances_anchor = self._distributed_sampling_anchor(bridge_instances,
                                                                        anchor=consts.HOST,
                                                                        target=consts.GUEST)
        LOGGER.info("display top 6 host_distributed_instances_anchor:")
        logDtableInstances(LOGGER, host_distributed_instances_anchor, topk=5, isInstance=False)

        guest_distributed_instances_target = self._distributed_sampling_target(bridge_instances,
                                                                         anchor=consts.GUEST,
                                                                         target=consts.HOST)
        LOGGER.info("display top 6 guest_distributed_instances_target:")
        logDtableInstances(LOGGER, guest_distributed_instances_target, topk=5, isInstance=False)

        # distributed negative neighbors sampling
        host_neg_instances_anchor = self._distributed_negative_sampling_anchor(host_distributed_instances_anchor, anchor=consts.HOST, target=consts.GUEST)
        
        guest_neg_instances_target = self._distributed_negative_sampling_target(adj_instances,
                                                                          anchor=consts.GUEST,
                                                                          target=consts.HOST)

        distributed_instances_anchor = host_distributed_instances_anchor.union(host_neg_instances_anchor)
        LOGGER.info("distributed_instances_target count: {}".format(distributed_instances_anchor.count()))

        distributed_instances_target = guest_distributed_instances_target.union(guest_neg_instances_target)
        LOGGER.info("distributed_instances_target count: {}".format(distributed_instances_target.count()))

        logDtableInstances(LOGGER, host_distributed_instances_anchor, topk=5, isInstance=False)
        logDtableInstances(LOGGER, host_neg_instances_anchor, topk=5, isInstance=False)

        return distributed_instances_target, distributed_instances_anchor

       
       
        



