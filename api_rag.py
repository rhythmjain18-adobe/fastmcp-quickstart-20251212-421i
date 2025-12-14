# -*- coding: utf-8 -*-
"""
RAG Database for InDesign API Documentation

Creates a vector database from the OpenAPI spec for accurate payload generation.

Usage:
    # Build the database (first time only)
    python api_rag.py --build
    
    # Query the database
    python api_rag.py --query "How do I create a PDF rendition?"
    
    # Generate a payload
    python api_rag.py --generate "create PDF from InDesign file"
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
import re

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    print("WARNING: ChromaDB not installed. Install with: uv pip install chromadb")
    chromadb = None

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, lightweight model
except ImportError:
    print("WARNING: sentence-transformers not installed. Install with: uv pip install sentence-transformers")
    SentenceTransformer = None


class InDesignAPIRAG:
    """RAG database for InDesign API documentation."""
    
    def __init__(self, spec_path: str = "indesignapi.json", db_path: str = ".chromadb"):
        self.spec_path = spec_path
        self.db_path = db_path
        self.collection_name = "indesign_api_docs"
        
        if chromadb and SentenceTransformer:
            self.client = chromadb.PersistentClient(path=db_path)
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            self.collection = None
        else:
            self.client = None
            self.embedding_model = None
    
    def _load_spec(self) -> Dict[str, Any]:
        """Load the OpenAPI specification."""
        with open(self.spec_path, 'r') as f:
            return json.load(f)
    
    def _chunk_documentation(self, spec: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Break down the API spec into meaningful chunks."""
        chunks = []
        
        # 1. Extract endpoint information
        for path, methods in spec.get("paths", {}).items():
            for method, details in methods.items():
                chunk = {
                    "type": "endpoint",
                    "path": path,
                    "method": method.upper(),
                    "summary": details.get("summary", ""),
                    "description": details.get("description", ""),
                    "operationId": details.get("operationId", ""),
                    "tags": details.get("tags", []),
                    "parameters": details.get("parameters", []),
                    "requestBody": details.get("requestBody", {}),
                    "content": self._format_endpoint_doc(path, method, details)
                }
                chunks.append(chunk)
        
        # 2. Extract schema information
        for schema_name, schema_def in spec.get("components", {}).get("schemas", {}).items():
            chunk = {
                "type": "schema",
                "name": schema_name,
                "description": schema_def.get("description", ""),
                "properties": schema_def.get("properties", {}),
                "required": schema_def.get("required", []),
                "content": self._format_schema_doc(schema_name, schema_def)
            }
            chunks.append(chunk)
        
        # 3. Extract general information
        info = spec.get("info", {})
        chunks.append({
            "type": "info",
            "title": info.get("title", ""),
            "description": info.get("description", ""),
            "content": f"Title: {info.get('title', '')}\n\nDescription: {info.get('description', '')}"
        })
        
        # 4. Extract tag descriptions
        for tag in spec.get("tags", []):
            chunk = {
                "type": "tag",
                "name": tag.get("name", ""),
                "description": tag.get("description", ""),
                "content": f"Tag: {tag.get('name', '')}\n\nDescription: {tag.get('description', '')}"
            }
            chunks.append(chunk)
        
        return chunks
    
    def _format_endpoint_doc(self, path: str, method: str, details: Dict) -> str:
        """Format endpoint documentation as searchable text."""
        doc = f"Endpoint: {method.upper()} {path}\n"
        doc += f"Operation: {details.get('operationId', '')}\n"
        doc += f"Summary: {details.get('summary', '')}\n"
        doc += f"Description: {details.get('description', '')}\n"
        doc += f"Tags: {', '.join(details.get('tags', []))}\n\n"
        
        # Add parameter info
        if details.get('parameters'):
            doc += "Parameters:\n"
            for param in details['parameters']:
                required = "required" if param.get('required') else "optional"
                doc += f"  - {param.get('name')} ({param.get('in')}, {required}): {param.get('description', '')}\n"
        
        # Add request body info
        if details.get('requestBody'):
            doc += "\nRequest Body:\n"
            rb = details['requestBody']
            doc += f"  Description: {rb.get('description', '')}\n"
            doc += f"  Required: {rb.get('required', False)}\n"
            
            # Extract schema reference if available
            content = rb.get('content', {})
            for content_type, content_details in content.items():
                doc += f"  Content-Type: {content_type}\n"
                if '$ref' in content_details.get('schema', {}):
                    schema_ref = content_details['schema']['$ref']
                    doc += f"  Schema: {schema_ref.split('/')[-1]}\n"
        
        return doc
    
    def _format_schema_doc(self, name: str, schema: Dict) -> str:
        """Format schema documentation as searchable text."""
        doc = f"Schema: {name}\n"
        doc += f"Description: {schema.get('description', '')}\n"
        
        if schema.get('required'):
            doc += f"Required fields: {', '.join(schema['required'])}\n"
        
        doc += "\nProperties:\n"
        for prop_name, prop_details in schema.get('properties', {}).items():
            prop_type = prop_details.get('type', 'unknown')
            prop_desc = prop_details.get('description', '')
            doc += f"  - {prop_name} ({prop_type}): {prop_desc}\n"
            
            # Add enum values if present
            if 'enum' in prop_details:
                doc += f"    Allowed values: {', '.join(map(str, prop_details['enum']))}\n"
            
            # Add default value if present
            if 'default' in prop_details:
                doc += f"    Default: {prop_details['default']}\n"
        
        return doc
    
    def build_database(self):
        """Build the RAG database from the API spec."""
        if not chromadb or not SentenceTransformer:
            print("ERROR: ChromaDB and sentence-transformers are required to build the database")
            return
        
        print("Loading API specification...")
        spec = self._load_spec()
        
        print("Chunking documentation...")
        chunks = self._chunk_documentation(spec)
        print(f"   Created {len(chunks)} documentation chunks")
        
        print("Generating embeddings...")
        # Extract text content and metadata
        documents = [chunk["content"] for chunk in chunks]
        metadatas = [{k: str(v) for k, v in chunk.items() if k != "content"} for chunk in chunks]
        ids = [f"{i}_{chunk.get('type', 'unknown')}" for i, chunk in enumerate(chunks)]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(documents, show_progress_bar=True)
        
        print("Storing in database...")
        # Delete existing collection if it exists
        try:
            self.client.delete_collection(self.collection_name)
        except:
            pass
        
        # Create new collection
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "InDesign API documentation"}
        )
        
        # Add documents in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            end_idx = min(i + batch_size, len(documents))
            self.collection.add(
                documents=documents[i:end_idx],
                embeddings=embeddings[i:end_idx].tolist(),
                metadatas=metadatas[i:end_idx],
                ids=ids[i:end_idx]
            )
        
        print(f"SUCCESS: Database built successfully!")
        print(f"   Stored {len(documents)} documents in {self.db_path}")
    
    def query(self, query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Query the RAG database."""
        if not self.collection:
            self.collection = self.client.get_collection(self.collection_name)
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query_text])[0]
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i] if 'distances' in results else None
            })
        
        return formatted_results
    
    def generate_payload(self, user_request: str) -> Dict[str, Any]:
        """Generate API payload based on user request using RAG."""
        print(f"\nAnalyzing request: '{user_request}'")
        
        # Query the database for relevant documentation
        results = self.query(user_request, n_results=3)
        
        print(f"Found {len(results)} relevant documentation sections\n")
        
        # Extract context
        context = "\n\n".join([r["content"] for r in results])
        
        # Determine the endpoint
        endpoint_info = None
        for r in results:
            if r["metadata"].get("type") == "endpoint":
                endpoint_info = r["metadata"]
                break
        
        if not endpoint_info:
            return {
                "error": "Could not determine the appropriate API endpoint",
                "context": context
            }
        
        # Build payload template based on endpoint
        payload = self._build_payload_template(endpoint_info, user_request)
        
        return {
            "endpoint": f"{endpoint_info.get('method')} {endpoint_info.get('path')}",
            "payload": payload,
            "documentation": context[:500] + "..." if len(context) > 500 else context
        }
    
    def _build_payload_template(self, endpoint_info: Dict, user_request: str) -> Dict[str, Any]:
        """Build a payload template for the detected endpoint."""
        operation_id = endpoint_info.get("operationId", "")
        
        # Template for create-rendition
        if operation_id == "renditionJob":
            return {
                "assets": [
                    {
                        "source": {
                            "url": "<presigned-url-to-indesign-file>",
                            "storageType": "AWS"  # or Azure, Dropbox
                        },
                        "destination": "document.indd"
                    }
                ],
                "params": {
                    "targetDocuments": ["document.indd"],
                    "outputMediaType": "application/pdf",  # or image/jpeg, image/png
                    "resolution": 72,
                    "pageRange": "All",
                    "quality": "medium"  # low, medium, high, maximum
                }
            }
        
        # Template for data merge
        elif operation_id == "dataMerge":
            return {
                "assets": [
                    {
                        "source": {
                            "url": "<presigned-url-to-template>",
                            "storageType": "AWS"
                        },
                        "destination": "template.indd"
                    },
                    {
                        "source": {
                            "url": "<presigned-url-to-csv>",
                            "storageType": "AWS"
                        },
                        "destination": "data.csv"
                    }
                ],
                "params": {
                    "targetDocument": "template.indd",
                    "dataSource": "data.csv",
                    "outputMediaType": "application/pdf",  # or image/jpeg, image/png, application/x-indesign
                    "recordRange": "All"
                }
            }
        
        # Template for document info
        elif operation_id == "getDocumentInfo":
            return {
                "assets": [
                    {
                        "source": {
                            "url": "<presigned-url-to-document>",
                            "storageType": "AWS"
                        },
                        "destination": "document.indd"
                    }
                ],
                "params": {
                    "targetDocument": "document.indd",
                    "pageInfo": {"enabled": True},
                    "linkInfo": {"enabled": True},
                    "fontInfo": {"enabled": True},
                    "pageItemInfo": {"enabled": False},
                    "textStoryInfo": {"enabled": False}
                }
            }
        
        # Template for remap links
        elif operation_id == "remapLinks":
            return {
                "assets": [
                    {
                        "source": {
                            "url": "<presigned-url-to-document>",
                            "storageType": "AWS"
                        },
                        "destination": "document.indd"
                    }
                ],
                "params": {
                    "targetDocument": "document.indd",
                    "dataSource": [
                        {
                            "sourceURI": "file:///path/to/old/image.jpg",
                            "destinationURI": "https://aem-server.com/assets/new-image.jpg"
                        }
                    ]
                }
            }
        
        # Template for data merge tags
        elif operation_id == "dataMergeTags":
            return {
                "assets": [
                    {
                        "source": {
                            "url": "<presigned-url-to-template>",
                            "storageType": "AWS"
                        },
                        "destination": "template.indd"
                    }
                ],
                "params": {
                    "targetDocument": "template.indd",
                    "filter": ["all"],  # or ["text", "image", "qr"]
                    "includePageItemIdentifiers": False
                }
            }
        
        else:
            return {
                "note": f"Template for {operation_id} - please refer to documentation"
            }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="InDesign API RAG Database")
    parser.add_argument("--build", action="store_true", help="Build the RAG database")
    parser.add_argument("--query", type=str, help="Query the database")
    parser.add_argument("--generate", type=str, help="Generate API payload from description")
    parser.add_argument("--spec", type=str, default="indesignapi.json", help="Path to OpenAPI spec")
    parser.add_argument("--db", type=str, default=".chromadb", help="Path to ChromaDB storage")
    
    args = parser.parse_args()
    
    rag = InDesignAPIRAG(spec_path=args.spec, db_path=args.db)
    
    if args.build:
        print("=" * 60)
        print("Building RAG Database")
        print("=" * 60)
        rag.build_database()
    
    elif args.query:
        print("=" * 60)
        print("Querying Database")
        print("=" * 60)
        results = rag.query(args.query)
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"Type: {result['metadata'].get('type', 'unknown')}")
            print(f"Distance: {result.get('distance', 'N/A'):.4f}")
            print(f"\nContent:\n{result['content'][:300]}...")
    
    elif args.generate:
        print("=" * 60)
        print("Generating API Payload")
        print("=" * 60)
        result = rag.generate_payload(args.generate)
        print(json.dumps(result, indent=2))
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

