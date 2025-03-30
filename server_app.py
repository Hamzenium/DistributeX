from flwr.server import ServerApp, ServerAppComponents, Context
from flwr.server.strategy import FedAvg
from flwr.common import ndarrays_to_parameters
from flwr.server import ServerConfig
from .task import load_model

def server_fn(context: Context) -> ServerAppComponents:
    num_rounds = context.run_config["num-server-rounds"]
    parameters = ndarrays_to_parameters(load_model().get_weights())
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)
