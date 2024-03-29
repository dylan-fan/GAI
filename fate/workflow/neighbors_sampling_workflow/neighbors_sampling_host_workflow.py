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

from arch.api.utils import log_utils
from federatedml.param import NeighborsSamplingParam
from federatedml.util.param_extract import ParamExtract
from workflow.workflow import WorkFlow
from federatedml.util import consts
from federatedml.neighbors_sampling.neighbors_sampling_host import NeighborsSamplingHost

LOGGER = log_utils.getLogger()


class NeighborsSamplingHostWorkFlow(WorkFlow):
    def _initialize_model(self, config_path):
        neighbos_sampling_param = NeighborsSamplingParam()
        self.neighbos_sampling_param = ParamExtract.parse_param_from_config(neighbos_sampling_param, config_path)
        self.neighbors_sampler = NeighborsSamplingHost(self.neighbos_sampling_param)

    def _initialize_role_and_mode(self):
        self.role = consts.HOST

if __name__ == "__main__":
    host_wf = NeighborsSamplingHostWorkFlow()
    host_wf.run()
