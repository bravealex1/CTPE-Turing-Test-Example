# To install: pip install tavily-python
from tavily import TavilyClient
client = TavilyClient("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")
response = client.search(
    query="Draw a cat"
)
print(response)