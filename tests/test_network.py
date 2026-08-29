from src.networks.biological_network import BiologicalNetwork


def test_network_propagation_affects_pathways():
    net = BiologicalNetwork.demo()
    state = net.propagate({"EGFR": 1.0})
    activity = net.pathway_activity(state)
    assert state["RAS"] > 0
    assert activity["MAPK"] > 0
