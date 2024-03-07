from dask.distributed import Client
from multiprocessing import freeze_support

if __name__ == '__main__':
    # Connect to the Dask client
    client = Client()
    freeze_support()
    # Shutdown all workers
    client.shutdown()
    # Close the client
    client.close()
