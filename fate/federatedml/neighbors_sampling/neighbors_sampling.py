import numpy as np
import math

from arch.api import federation
from arch.api import eggroll
from arch.api.utils import log_utils
from arch.api.utils.log_utils import logDtableInstances
from federatedml.secureprotol.encode import Encode
from federatedml.util import consts
from federatedml.util import NeighborsSamplingChecker
from federatedml.util.transfer_variable import RawIntersectTransferVariable
from federatedml.param import NeighborsSamplingParam
from federatedml.util.data_io import Instance
from federatedml.util import fate_operator, abnormal_detection
from federatedml.neighbors_sampling.discrete_distribution_sampling import DiscreteDistributionSampler

LOGGER = log_utils.getLogger()


LOGGER = log_utils.getLogger()

class NeighborsSampling(object):
    def __init__(self, neighbors_sampling_param: NeighborsSamplingParam):
        self.param = neighbors_sampling_param
        NeighborsSamplingChecker.check_param(neighbors_sampling_param)
        
        # set params
        self.times_of_sampling = neighbors_sampling_param.times_of_sampling
        self.w = neighbors_sampling_param.w
        self.nega_samp_num = neighbors_sampling_param.nega_samp_num
        self.transfer_variable = None
    
    @staticmethod
    def random_walk(start_instance, data_instances, times_of_sampling, w):
        """ 
        Random walk from the node which has the adjlist
        
        Parameters
        ----------
        start_instance: data_io.Instance. This is the adjlist instance of the start node.

        data_instance: DTable of data_io.Instance. A object which is composed with adjlists Instance of each nodes.

        w: int. The length of random walk.
        """
        paths = []
        for _ in range(times_of_sampling):
            cur = start_instance.features[np.random.randint(start_instance.features.shape[0])]
            path = []
            while len(path) < w:
                path.append(cur)
                id_cur = str(cur)
                instance = data_instances.get(id_cur)
                neis = instance.features
                node = neis[np.random.randint(neis.shape[0])]
                cur = node
            paths.append(path)
        return Instance(features=paths)
    
    @staticmethod
    def negative_sampling(k, v, sampler, distribution, nega_samp_num):
        """
        distribution: list of tuples. [(node1(int), p1), ...]
        """
        negative_pairs = []
        anchor = v[0]
        for i in range(nega_samp_num):
            index = sampler.sampling()
            neg_target = distribution[index][0]
            pair = (k + '_negative_' + str(i), (anchor, neg_target, -1))
            negative_pairs.append(pair)
        
        return negative_pairs
    
    @staticmethod
    def generate_nega_distribution(adj_instances):
        """
        adj_instances: DTable of Instances. Node: Instance(sid=None, features=adj)
        """
        degree_instances = adj_instances.mapValues(lambda v: math.pow(len(v.features), 0.75))          # pow 3/4
        degree_sum = degree_instances.reduce(lambda x, y: x + y)
        distribution_instance = degree_instances.mapValues(lambda v: v / degree_sum) 
        
        distribution = list(distribution_instance.map(lambda k, v: (int(k), v)).collect())                        # [(node1, p1),..., (noden, pn)]
        distribution = sorted(distribution, key=lambda v: v[0])
        
        return distribution

    def local_negative_sampling(self, adj_instances, positive_instances):
        """
        Sampling several negativae samples according to the distribution of anchor node's degree for each positive samples.
        Parameters
        ----------
        adj_instances: Dtable of Instance which has adjlist for each node.
        positive_instances: Dtable of positive samples(pairs). 
        """
        
        distribution = NeighborsSampling.generate_nega_distribution(adj_instances)

        sampler = DiscreteDistributionSampler([data[1] for data in distribution])
        negative_instances = positive_instances.flatMap(lambda k, v: NeighborsSampling.negative_sampling(k, v, sampler, distribution, self.nega_samp_num))
        logDtableInstances(LOGGER, negative_instances, topk=10, isInstance=False)

        LOGGER.info("distribution len: {}, max:{}".format(len(distribution), max(distribution, key=lambda v: v[1])))
       
        return negative_instances
        


    def local_neighbors_sampling(self, data_instances, role):
        # positive instances sampled
        walk_instances = data_instances.mapValues(lambda v: NeighborsSampling.random_walk(v, data_instances, self.times_of_sampling, self.w))
        
        # unpack all walk sequence of a anchor node to the context list
        def unpack_walks(walk_instance):
            walks = walk_instance.features
            neighbors = [context for walk in walks for context in walk]   

            return neighbors
        
        def construct_pairs(anchor, targets):
            pairs = []
            for index, target in enumerate(targets):
                if int(anchor) != target:
                    pairs.append((role + '_' + anchor + '_' + str(index), (int(anchor), target, 1))) # type(anchor(key)) = str, transform it to int
            
            return pairs

        neighbors_instances = walk_instances.mapValues(lambda walk_instance: unpack_walks(walk_instance))
        positive_instances = neighbors_instances.flatMap(lambda anchor, targets: construct_pairs(anchor, targets))
        logDtableInstances(LOGGER, positive_instances, topk=10, isInstance=False)
        LOGGER.info("The number of local positive pairs: {}".format(positive_instances.count()))

        # negative instances sampled
        LOGGER.info("Negative samples num: {}".format(self.nega_samp_num))
        negative_instances = self.local_negative_sampling(data_instances, positive_instances)
        LOGGER.info("The number of local negative pairs: {}".format(negative_instances.count()))

        local_instances = positive_instances.union(negative_instances)
        

        logDtableInstances(LOGGER, positive_instances, topk=3, isInstance=False)
        logDtableInstances(LOGGER, negative_instances, topk=3, isInstance=False)

        LOGGER.info("The number of total local pairs: {}".format(local_instances.count()))

        return local_instances



    def distributed_neighbors_sampling(self, data_instances):
        pass

    @staticmethod
    def find_exclusive_neis(instance, common_node_ids):
        """
        Find the neighoor nodes which are not in common nodes for common nodes.
        """
        neis = instance.features
        new_neis = [nei for nei in neis if str(nei) not in common_node_ids]

        return Instance(features=new_neis)
    
    @staticmethod
    def get_bridge_nodes(common_instances):
        """
        Get the possible bridge nodes, but not be alignment with each other.
        """
        common_node_ids = [data[0] for data in common_instances.collect()]
        bridge_instances = common_instances.mapValues(lambda v: NeighborsSampling.find_exclusive_neis(v, common_node_ids))
        bridge_instances = bridge_instances.filter(lambda k, v: len(v.features) > 0)
        return bridge_instances

    @staticmethod
    def gen_distributed_sample_ids(bridge_instances, number_instances, role):
        """
        Generate samples which are distributed accross two parties
        Parameters
        ----------
        number: int. The number of individual neighbors of opposite party
        """
        def gen_ids(v, number):
            neis = v.features
            sample_ids = []
            for i in range(len(neis)):
                for j in range(number):
                    sample_ids.append(('_'  + str(i) + '_' + str(j), neis[i]))

            return Instance(features=sample_ids)

        def add_prefix(k, v, role):
            samples = v.features
            new_samples = [(role + '_' + k + sample[0], sample[1]) for sample in samples]
            return (k, Instance(features=new_samples))

        def extract_id(v):
            samples = v.features
            ids = [sample[0] for sample in samples]
            return Instance(features=ids)

        bridge_instances = bridge_instances.join(number_instances, lambda v, number: gen_ids(v, number))
        LOGGER.info("bridge_instances count:{}".format(bridge_instances.count()))

        bridge_instances = bridge_instances.map(lambda k, v: add_prefix(k, v, role))

        sample_ids_instances = bridge_instances.mapValues(extract_id)

        distributed_instances = bridge_instances.flatMap(lambda k, v: v.features)
      
        LOGGER.info("sample_ids_instances count:{}".format(sample_ids_instances.count()))

        return sample_ids_instances, distributed_instances

    @staticmethod
    def gen_distributed_instances(bridge_instances, sample_ids_instances):
        """
        Generate the 2-hop neighbors instance with the nodes distributed accross two parties. Allocate the sample_ids to the nei nodes in bridge_instances.
        """
        def allocate(v1, v2):
            nodes = v1.features
            sample_ids = v2.features
            samples = []
            node_num = len(nodes)

            for index, id in enumerate(sample_ids):
                # positive samples
                samples.append((id, (nodes[index % node_num], 1)))

            return Instance(features=samples)

        bridge_instances = bridge_instances.join(sample_ids_instances, lambda v1, v2: allocate(v1, v2))

        return bridge_instances.flatMap(lambda k, v: v.features)

    def _distributed_sampling_anchor(self, bridge_instances, anchor=consts.HOST, target=consts.GUEST):

        if anchor == consts.HOST:
            if target != consts.GUEST:
                raise NameError("if anchor is host, then target should be guest!!!")
            number_transfer = self.transfer_variable.guest_number
            sample_id_transfer = self.transfer_variable.host_sample_ids
        elif anchor == consts.GUEST:
            if target != consts.HOST:
                raise NameError("if anchor is guest, then target should be host!!!")
            number_transfer = self.transfer_variable.host_number
            sample_id_transfer = self.transfer_variable.guest_sample_ids
        else:
            raise NameError("anchor should be choose from {host, guest}")
        
        LOGGER.info("Generate the distributed {} samples for {}".format(anchor, anchor))
        number_instances = federation.get(name=number_transfer.name,
                                 tag=self.transfer_variable.generate_transferid(number_transfer),
                                 idx=0)
        LOGGER.info("Get numbers from {}".format(target))
        #logDtableInstances(LOGGER, number_instances, topk=5, isInstance=False)

        sample_id_instances, distributed_instances = NeighborsSampling.gen_distributed_sample_ids(bridge_instances, number_instances, anchor)
        LOGGER.info("distributed_instances count:{}".format(distributed_instances.count()))

        federation.remote(sample_id_instances,
                          name=sample_id_transfer.name,
                          tag=self.transfer_variable.generate_transferid(sample_id_transfer),
                          role=target,
                          idx=0)
        LOGGER.info("Remote generated distributed samples' id to {}".format(target))
        
        # distributed_instances_anchor (the input node in skip-gram which is the part of sample)

        return distributed_instances
    

    def _distributed_sampling_target(self, bridge_instances, anchor=consts.HOST, target=consts.GUEST):

        if anchor == consts.HOST:
            if target != consts.GUEST:
                raise NameError("if anchor is host, then target should be guest!!!")
            number_transfer = self.transfer_variable.guest_number
            sample_id_transfer = self.transfer_variable.host_sample_ids
        elif anchor == consts.GUEST:
            if target != consts.HOST:
                raise NameError("if anchor is guest, then target should be host!!!")
            number_transfer = self.transfer_variable.host_number
            sample_id_transfer = self.transfer_variable.guest_sample_ids
        else:
            raise NameError("anchor should be choose from {host, guest}")
        
        LOGGER.info("Generate the distributed {} samples for {}".format(anchor, target))
        number_instances = bridge_instances.mapValues(lambda v: len(v.features)) 
        federation.remote(number_instances,
                          name=number_transfer.name,
                          tag=self.transfer_variable.generate_transferid(number_transfer),
                          role=anchor,
                          idx=0)
        LOGGER.info("Remote numbers to {}".format(anchor))

        sample_id_instances = federation.get(name=sample_id_transfer.name,
                                             tag=self.transfer_variable.generate_transferid(sample_id_transfer),
                                             idx=0)
        LOGGER.info("Get generated distributed samples' id from {}".format(anchor))
        #logDtableInstances(LOGGER, sample_id_instances, 5)

        distributed_instances =  NeighborsSampling.gen_distributed_instances(bridge_instances, sample_id_instances)
        #logDtableInstances(LOGGER, distributed_instances, 5, isInstance=False)
        LOGGER.info("The number of {} distributed_instances:{}".format(anchor, distributed_instances.count()))

        # distributed_instances_target (to be predicted node in skip-gram which is the part of sample)
        return distributed_instances
   
    def _distributed_negative_sampling_anchor(self, positive_instances, anchor=consts.HOST, target=consts.GUEST):

        if anchor == consts.HOST:
            if target != consts.GUEST:
                raise NameError("if anchor is host, then target should be guest!!!")
            nega_ids_transfer = self.transfer_variable.host_neg_samp_ids
        elif anchor == consts.GUEST:
            if target != consts.HOST:
                raise NameError("if anchor is guest, then target should be host!!!")
            nega_ids_transfer = self.transfer_variable.guest_neg_samp_ids
        else:
            raise NameError("anchor should be choose from {host, guest}")

        def gen_neg_ids(k, v, neg_sum):
            ids = []
            for i in range(neg_sum):
                ids.append((k + '_negative_' + str(i), v))
            
            return (k, ids)

        distributed_negative_instances_anchor = positive_instances.map(lambda k, v: gen_neg_ids(k, v, self.nega_samp_num)).flatMap(lambda k, v: v)
        distributed_negative_ids = distributed_negative_instances_anchor.take(distributed_negative_instances_anchor.count(), keysOnly=True)

        federation.remote(distributed_negative_ids,
                          name=nega_ids_transfer.name,
                          tag=self.transfer_variable.generate_transferid(nega_ids_transfer),
                          role=target,
                          idx=0)
        LOGGER.info("Remote the distributed negative instances id to {} from {}".format(target, anchor))
        LOGGER.info("Distributed negative instances count: {}".format(len(distributed_negative_ids)))
        logDtableInstances(LOGGER, distributed_negative_instances_anchor, topk=10, isInstance=False)

        return distributed_negative_instances_anchor


    def _distributed_negative_sampling_target(self, adj_instances, anchor=consts.HOST, target=consts.GUEST):
        if anchor == consts.HOST:
            if target != consts.GUEST:
                raise NameError("if anchor is host, then target should be guest!!!")
            nega_ids_transfer = self.transfer_variable.host_neg_samp_ids
        elif anchor == consts.GUEST:
            if target != consts.HOST:
                raise NameError("if anchor is guest, then target should be host!!!")
            nega_ids_transfer = self.transfer_variable.guest_neg_samp_ids
        else:
            raise NameError("anchor should be choose from {host, guest}")

        distributed_negative_ids = federation.get(name=nega_ids_transfer.name,
                                                  tag=self.transfer_variable.generate_transferid(nega_ids_transfer),
                                                  idx=0)
        LOGGER.info("Get distributed nagative samples from {}".format(anchor))
        for i in range(10):
            LOGGER.info("id:{}".format(distributed_negative_ids[i]))

        #sample some negative samples
        distribution = NeighborsSampling.generate_nega_distribution(adj_instances)
        sampler = DiscreteDistributionSampler([data[1] for data in distribution])

        distributed_negative_instances_target = eggroll.table(name=target + eggroll.generateUniqueId(),
                                                           namespace='neighbors_sampling/distributed_sampling',
                                                           persistent=False)
                                                           
        for id in distributed_negative_ids:
            index = sampler.sampling()
            distributed_negative_instances_target.put(id, (distribution[index][0], -1))
        
        logDtableInstances(LOGGER, distributed_negative_instances_target, isInstance=False)

        return distributed_negative_instances_target
                
        
    def set_header(self, header):
        self.header = header
    
    def get_header(self, data_instances):
        if self.header is not None:
            return self.header
        
        return data_instances.schema.get("header")

    def set_flowid(self, flowid=0):
        if self.transfer_variable is not None:
            self.transfer_variable.set_flowid(flowid)
            LOGGER.debug("set flowid:" + str(flowid))

    def _abnormal_detection(self, data_instances):
        """
        Make sure input data_instance is valid
        """
        abnormal_detection.empty_table_detection(data_instances)
        abnormal_detection.empty_feature_detection(data_instances)

