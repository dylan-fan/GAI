from federatedml.network_embedding.hetero_network_embedding import HeteroNEGuest
from federatedml.param import NetworkEmbeddingParam
from federatedml.util import consts
from federatedml.util import ParamExtract
from workflow.workflow import WorkFlow
from arch.api.utils import log_utils

LOGGER = log_utils.getLogger()

class NEGuestWorkFLow(WorkFlow):
    def _initialize_model(self, config):
        network_embedding_param = NetworkEmbeddingParam()
        self.network_embedding_param = ParamExtract.parse_param_from_config(network_embedding_param, config)
        self.nrler = HeteroNEGuest(self.network_embedding_param)

    def _initialize_role_and_mode(self):
        self.role = consts.GUEST
        self.mode = consts.HETERO

if __name__ == '__main__':
    guest_wf = NEGuestWorkFLow()
    guest_wf.run()
