from arch.api import eggroll
from arch.api.utils.log_utils import logDtableInstances, getLogger
import numpy as np

eggroll.init(mode=0)
LOGGER = getLogger()

def cos_sim(a, b):
    inner_product = a.dot(b)
    div = np.sqrt(a.dot(a)) * np.sqrt(b.dot(b))

    return inner_product / div

def show_local_samples(name, namespace, topk=5):
    local_samples = eggroll.table(name, namespace, persistent=True)
    samples = list(local_samples.collect())
    for data in samples[: topk]:
        print("sample_id: {}, training pairs:{}".format(data[0], data[1]))
    
    for data in samples[-topk: ]:
        print("sample_id: {}, training pairs:{}".format(data[0], data[1]))

show_local_samples('host', 'neighbors_samples/local_samples')

def show_distributed_samples(topk):
    samples_anchor = eggroll.table('anchor', "neighbors_samples/distributed_samples/host", persistent=True)
    samples_target = eggroll.table('target', "neighbors_samples/distributed_samples/guest", persistent=True)

    samples_anchor = list(samples_anchor.collect())
    samples_target = list(samples_target.collect())

    for anchor, target in zip(samples_anchor[: topk + 10], samples_target):
        print("sample_id: {}, anchor:{}   sample_id: {}, target:{}".format(anchor[0], anchor[1], target[0], target[1]))
    
show_distributed_samples(20)

def show_embedding():
    host_embedding = eggroll.table('host', 'node_embedding', persistent=True)
    guest_embedding = eggroll.table('guest', 'node_embedding', persistent=True)
    print(guest_embedding.count())
    common_nodes = eggroll.table("common_nodes", "common_nodes", persistent=True)
    common_nodes = common_nodes.take(common_nodes.count(), keysOnly=True)
    for node in common_nodes[0:5]:
        node = int(node)
        sim = cos_sim(host_embedding.get(node), guest_embedding.get(node))
        print("node: {}, sim: {}".format(node, sim))
    
    print(cos_sim(host_embedding.get(8), guest_embedding.get(12)))

    """
    print("Bank A learning results:")
    host_embedding = list(host_embedding.collect())
    for data in host_embedding[0:10]:
        print("node id:{}, embedding: {}".format(data[0], data[1]))

    print()
    print()

    print("Bank B learning results:")
    guest_embedding = list(guest_embedding.collect())
    for data in guest_embedding[0:10]:
        print("node id:{}, embedding: {}".format(data[0], data[1]))
    """
show_embedding()

    
