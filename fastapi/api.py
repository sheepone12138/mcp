from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP
import asyncio
# Initialize FastMCP server
mcp = FastMCP("graphragapi")
# Constants
NWS_API_BASE = "http://127.0.0.1:8000/search/local"
async def make_graphragapi_request(url: str) -> dict[str, Any] | None:
    """
        访问网站
    """
    headers = {
        "Accept": "application/json"
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None
@mcp.tool()
async def get_fastadmin_kn(key: str) -> str:
    """Get fastadmin answer.

    Args:
        key: search the key from the fastadmin
    """
    # First get the forecast grid endpoint
    points_url =  f"{NWS_API_BASE}?query={key}"
    points_data = await make_graphragapi_request(points_url)

    if not points_data:
        return "Unable to fetch detailed forecast"
    response = points_data['response']
    return response
if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')
    # result = asyncio.run(get_fastadmin_kn('如何在列表操作列上增加一个按钮'))
    # if result:
    #     print(result)