"""
FastMCP InDesign Server

MCP server for Adobe InDesign API - create renditions from InDesign documents.

Usage:
    fastmcp run fastmcp_server.py
    fastmcp run fastmcp_server.py --transport sse --port 8000
"""

from fastmcp import FastMCP
import os
import json
import httpx
import re
from urllib.parse import urlsplit, unquote
from typing import Any

import boto3
from botocore.config import Config as BotoConfig
try:
    from env_loader import load_env_from_file  # optional helper
except Exception:
    def load_env_from_file() -> None:
        """No-op fallback if `env_loader` is not available."""
        return None

# Load environment variables (if helper provided)
load_env_from_file()

# Create server
mcp = FastMCP("InDesign API Server")

# API Configuration
ADOBE_API_BASE = "https://indesign.adobe.io"
ADOBE_API_VERSION = "v3"


# ============ Helper Functions ============

def _resolve_auth() -> dict[str, str]:
    """Resolve Adobe API authentication from environment."""
    api_key = os.getenv("ADOBE_API_KEY", "")
    access_token = os.getenv("ADOBE_ACCESS_TOKEN", "")
    ims_org_id = os.getenv("ADOBE_IMS_ORG_ID", "")

    headers: dict[str, str] = {
        "Authorization": f"Bearer {access_token}" if access_token else "",
        "x-api-key": api_key,
        "Content-Type": "application/json",
    }
    if ims_org_id:
        headers["x-gw-ims-org-id"] = ims_org_id
    return {k: v for k, v in headers.items() if v}


def _infer_storage_type(url: str) -> str:
    """Infer storage type from URL."""
    try:
        host = urlsplit(url).netloc.lower()
    except Exception:
        return "external"

    if host.endswith("amazonaws.com") or ".s3." in host:
        return "AWS"
    if ".blob.core.windows.net" in host:
        return "Azure"
    if "dropbox" in host:
        return "Dropbox"
    return "external"


def _sanitize_filename(name: str) -> str:
    """Sanitize a filename."""
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    sanitized = re.sub(r"_+", "_", sanitized).strip("._-")
    return sanitized or "document.indd"


async def _http_request(method: str, url: str, headers: dict, json_body: dict | None = None) -> dict | None:
    """Make HTTP request to Adobe API."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.request(method=method, url=url, headers=headers, json=json_body)
            resp.raise_for_status()
            return resp.json() if resp.text else {}
        except Exception as e:
            return {"error": str(e)}


# ============ MCP Tools ============

@mcp.tool
async def create_rendition(
    source_url: str,
    output_format: str,
    page_range: str = "All",
    resolution: int = 72,
    quality: str = "medium"
) -> str:
    """
    Create a PDF, JPEG, or PNG rendition from an InDesign document.
    
    Args:
        source_url: Pre-signed URL to the InDesign document (.indd or .idml)
        output_format: Output format - 'pdf', 'jpeg', or 'png'
        page_range: Pages to export - 'All', '1', '1-3', etc. (default: All)
        resolution: Resolution in DPI (default: 72)
        quality: Quality level - 'low', 'medium', 'high', 'maximum' (default: medium)
    
    Returns:
        JSON with job status URL to poll for completion
    """
    # Map format to media type
    format_map = {
        "pdf": "application/pdf",
        "jpeg": "image/jpeg", 
        "jpg": "image/jpeg",
        "png": "image/png"
    }
    output_media_type = format_map.get(output_format.lower(), output_format)
    
    headers = _resolve_auth()
    url = f"{ADOBE_API_BASE}/{ADOBE_API_VERSION}/create-rendition"
    
    # Get filename from URL
    try:
        basename = os.path.basename(urlsplit(source_url).path)
        target_doc = _sanitize_filename(unquote(basename))
    except Exception:
        target_doc = "document.indd"
    
    payload = {
        "assets": [{
            "source": {
                "url": source_url,
                "storageType": _infer_storage_type(source_url)
            },
            "destination": target_doc
        }],
        "params": {
            "targetDocuments": [target_doc],
            "outputMediaType": output_media_type,
            "resolution": resolution,
            "pageRange": page_range,
            "quality": quality
        }
    }
    
    data = await _http_request("POST", url, headers, payload)
    return json.dumps(data, indent=2)


@mcp.tool
async def get_rendition_status(status_url: str) -> str:
    """
    Check the status of a rendition job.
    
    Args:
        status_url: The status URL returned from create_rendition
    
    Returns:
        JSON with job status and output URLs when complete
    """
    headers = _resolve_auth()
    data = await _http_request("GET", status_url, headers)
    return json.dumps(data, indent=2)


@mcp.tool
async def upload_to_s3(file_path: str, expires_in: int = 3600) -> str:
    """
    Upload a local file to S3 and get a pre-signed URL.
    
    Args:
        file_path: Local path to the file to upload
        expires_in: URL expiration time in seconds (default: 3600)
    
    Returns:
        Pre-signed URL that can be used as source_url for create_rendition
    """
    import mimetypes
    
    bucket = os.getenv("S3_BUCKET") or os.getenv("AWS_S3_BUCKET")
    if not bucket:
        return json.dumps({"error": "S3_BUCKET not configured in environment"})
    
    if not os.path.exists(file_path):
        return json.dumps({"error": f"File not found: {file_path}"})
    
    key = os.path.basename(file_path)
    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
    
    try:
        s3 = boto3.client(
            "s3",
            region_name=region,
            config=BotoConfig(signature_version="s3v4")
        )
        
        content_type, _ = mimetypes.guess_type(file_path)
        extra_args = {"ContentType": content_type} if content_type else None
        
        if extra_args:
            s3.upload_file(file_path, bucket, key, ExtraArgs=extra_args)
        else:
            s3.upload_file(file_path, bucket, key)
        
        url = s3.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expires_in,
        )
        
        return json.dumps({
            "success": True,
            "presigned_url": url,
            "bucket": bucket,
            "key": key,
            "expires_in": expires_in
        }, indent=2)
        
    except Exception as e:
        return json.dumps({"error": str(e)})


# ============ MCP Resources ============

@mcp.resource("indesign://config")
def get_config() -> str:
    """Get current InDesign API configuration status."""
    return json.dumps({
        "api_base": ADOBE_API_BASE,
        "api_version": ADOBE_API_VERSION,
        "api_key_configured": bool(os.getenv("ADOBE_API_KEY")),
        "access_token_configured": bool(os.getenv("ADOBE_ACCESS_TOKEN")),
        "s3_bucket_configured": bool(os.getenv("S3_BUCKET") or os.getenv("AWS_S3_BUCKET")),
    }, indent=2)


@mcp.resource("indesign://formats")
def get_formats() -> str:
    """Get supported output formats."""
    return json.dumps({
        "formats": [
            {"name": "PDF", "media_type": "application/pdf", "extension": ".pdf"},
            {"name": "JPEG", "media_type": "image/jpeg", "extension": ".jpg"},
            {"name": "PNG", "media_type": "image/png", "extension": ".png"}
        ],
        "quality_options": ["low", "medium", "high", "maximum"],
        "page_range_examples": ["All", "1", "1-3", "1,3,5"]
    }, indent=2)


@mcp.resource("indesign://help/{topic}")
def get_help(topic: str) -> str:
    """Get help on a specific topic."""
    help_topics = {
        "create_rendition": """
Create Rendition Tool:
- source_url: Pre-signed URL to your .indd or .idml file
- output_format: 'pdf', 'jpeg', or 'png'
- page_range: 'All' for all pages, or specific pages like '1-3'
- resolution: DPI value (72, 150, 300 recommended)
- quality: 'low', 'medium', 'high', or 'maximum'
""",
        "status": """
Get Rendition Status:
- Use the status_url from create_rendition response
- Poll until status shows 'succeeded' or 'failed'
- Output URLs are in the response when complete
""",
        "upload": """
Upload to S3:
- Requires S3_BUCKET and AWS credentials in environment
- Returns a pre-signed URL valid for the specified time
- Use the URL as source_url for create_rendition
"""
    }
    return help_topics.get(topic, f"Unknown topic: {topic}. Available: {list(help_topics.keys())}")


# ============ MCP Prompts ============

@mcp.prompt("convert_to_pdf")
def convert_to_pdf_prompt(source_url: str) -> str:
    """Prompt to convert an InDesign document to PDF."""
    return f"""Please convert this InDesign document to PDF:

Source URL: {source_url}

Steps:
1. Call create_rendition with output_format='pdf'
2. Poll get_rendition_status until complete
3. Return the output PDF URL"""


@mcp.prompt("convert_to_images")
def convert_to_images_prompt(source_url: str, format: str = "jpeg") -> str:
    """Prompt to convert InDesign pages to images."""
    return f"""Please convert this InDesign document to {format.upper()} images:

Source URL: {source_url}
Format: {format}

Steps:
1. Call create_rendition with output_format='{format}'
2. Poll get_rendition_status until complete  
3. Return the output image URLs"""


if __name__ == "__main__":
    mcp.run()

