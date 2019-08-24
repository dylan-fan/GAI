from federatedml.network_embedding.hetero_network_embedding import HeteroNEHost
from federatedml.param import NetworkEmbeddingParam
from federatedml.util import consts
from federatedml.util import ParamExtract
from workflow.workflow import WorkFlow
from arch.api.utils import log_utils

LOGGER = log_utils.getLogger()

class NEHostWorkFlow(WorkFlow):
    def _initialize_model(self, config):
        network_embedding_param = NetworkEmbeddingParam()
        self.network_embedding_param = ParamExtract.parse_param_from_config(network_embedding_param, config)
        self.nrler = HeteroNEHost(self.network_embedding_param)

    def _initialize_role_and_mode(self):
        self.role = consts.HOST
        self.mode = consts.HETERO

if __name__ == '__main__':
    host_wf = NEHostWorkFlow()
    host_wf.run()