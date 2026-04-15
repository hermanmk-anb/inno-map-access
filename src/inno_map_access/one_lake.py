import rioxarray
import zarr
from obstore.auth.azure import AzureCredentialProvider
from obstore.store import AzureStore
from zarr.storage import ObjectStore

zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})
rioxarray.show_versions()
# 1. Configuration & Authentication
# Fabric OneLake URI: https://onelake.dfs.fabric.microsoft.com/<workspace>/<item>.<type>/<path>
WORKSPACE_ID = "23b1515f-7adb-4ee2-bebe-8e41b9f5f2d2"
LAKEHOUSE_ID = "eae09446-a6cd-40af-bd43-1d11616a3cac"


def get_onelake_zarr_store(
    workspace_id: str,
    lakehouse_id: str,
    read_only: bool = True,
) -> ObjectStore:
    """
    Create a Zarr-compatible ObjectStore backed by Azure OneLake storage.

    This function configures an AzureStore pointing to a specific path inside
    a Microsoft Fabric Lakehouse and wraps it in a Zarr ObjectStore for use
    with Zarr-based data access.

    Parameters
    ----------
    workspace_id : str
        The Microsoft Fabric workspace ID (used as the container name).
    lakehouse_id : str
        The Lakehouse ID within the workspace.
    fabric_geo_image_name : str
        The name of the dataset or folder under the Lakehouse "Files" directory.
    read_only : bool, optional
        If True, the returned store will be read-only. Default is True.

    Returns
    -------
    ObjectStore
        A Zarr-compatible object store backed by Azure OneLake.
    """

    # Initialize Azure credential provider (handles authentication)
    credential_provider = AzureCredentialProvider()

    azure_store = AzureStore(
        account_name="onelake",  # OneLake storage account
        container_name=workspace_id,  # Workspace acts as container
        prefix=f"{lakehouse_id}/Files/experiment2",  # Path to the data
        use_fabric_endpoint=True,  # Use Microsoft Fabric endpoint
        credential_provider=credential_provider,  # Authentication handler
    )
    # Wrap AzureStore in a Zarr ObjectStore interface
    return ObjectStore(azure_store, read_only=read_only)


ONELAKE_STORE = get_onelake_zarr_store(
    workspace_id=WORKSPACE_ID,
    lakehouse_id=LAKEHOUSE_ID,
    read_only=False,
)
