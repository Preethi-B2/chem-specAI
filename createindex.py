import os
from dotenv import load_dotenv
 
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
    SearchField,
    SearchFieldDataType
)
 
from azure.core.credentials import AzureKeyCredential
 
load_dotenv()
 
endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
key = os.getenv("AZURE_SEARCH_API_KEY")
index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
 
credential = AzureKeyCredential(key)
 
client = SearchIndexClient(endpoint, credential)
 
 
fields = [
 
    SimpleField(name="id", type=SearchFieldDataType.String, key=True),
 
    SearchableField(
        name="content",
        type=SearchFieldDataType.String
    ),
 
    SimpleField(
        name="section",
        type=SearchFieldDataType.String,
        filterable=True
    ),
 
    SimpleField(
        name="type",
        type=SearchFieldDataType.String,
        filterable=True
    ),
 
    SimpleField(
        name="source",
        type=SearchFieldDataType.String
    ),
 
    SearchField(
        name="contentVector",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        vector_search_dimensions=1536,
        vector_search_profile_name="vector-profile"
    )
]
 
 
vector_search = VectorSearch(
    profiles=[
        VectorSearchProfile(
            name="vector-profile",
            algorithm_configuration_name="vector-config"
        )
    ],
    algorithms=[
        HnswAlgorithmConfiguration(
            name="vector-config"
        )
    ]
)
 
 
index = SearchIndex(
    name=index_name,
    fields=fields,
    vector_search=vector_search
)
 
 
client.create_index(index)
 
print("Index created successfully")