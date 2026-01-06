from master_mcp_client import MCPClient
import asyncio
import time

async def main():
    client = MCPClient()
    
    await client.connect_to_server("python", ["master_mcp_client.py"])

    # list tools
    t = await client.session.list_tools()
    print(t)
    # test tool calling
    r = await client('chart-generate_area_chart', {'data': [{"time": "0", "value": 0}, {"time": "1", "value": 0}]})
    print(r)
    await client.cleanup()
    print("FINISHED")


asyncio.run(main())