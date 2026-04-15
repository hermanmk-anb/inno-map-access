from azure.identity import DefaultAzureCredential
from obstore.store import AzureStore

# 2. Fabric
WORKSPACE_ID = "23b1515f-7adb-4ee2-bebe-8e41b9f5f2d2"
LAKEHOUSE_ID = "eae09446-a6cd-40af-bd43-1d11616a3cac"

# Credentials
TOKEN_SCOPE: str = "https://storage.azure.com/.default"


def get_azure_store():
    """Create a azur zarr store."""
    TOKEN = DefaultAzureCredential().get_token(TOKEN_SCOPE).token
    azure_store = AzureStore(
        account_name="onelake",
        container_name=WORKSPACE_ID,
        prefix=f"{LAKEHOUSE_ID}/Files/sat_experiment/",
        use_fabric_endpoint=True,
        token=TOKEN,
    )
    return azure_store


store = get_azure_store()
