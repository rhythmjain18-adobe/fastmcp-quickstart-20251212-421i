# -*- coding: utf-8 -*-
"""
Intelligent InDesign MCP Server

A RAG-powered MCP server that automatically determines which InDesign API to use
based on natural language prompts and executes them.

Supported APIs:
- Rendition (PDF, JPEG, PNG)
- Document Info (fonts, links, pages)
- Data Merge (CSV + template)
- Data Merge Tags (extract tags)
- Remap Links (file to AEM URLs)
- Custom Scripts (advanced automation)

Built with Adobe InDesign API best practices from official guides:
https://developer.adobe.com/firefly-services/docs/indesign-apis/guides/

Usage:
    fastmcp run intelligent_server.py
    fastmcp run intelligent_server.py --transport sse --port 8000
"""

from fastmcp import FastMCP
import os
import json
import httpx
import re
import asyncio
from urllib.parse import urlsplit, unquote
from typing import Any, Dict, Optional

import boto3
from botocore.config import Config as BotoConfig
# from env_loader import load_env_from_file

# # Load environment variables
# load_env_from_file()

# Create server
mcp = FastMCP("Intelligent InDesign API Server")

# API Configuration
ADOBE_API_BASE = "https://indesign.adobe.io"
ADOBE_API_VERSION = "v3"

# API Limits (from Adobe Technical Usage documentation)
# Reference: https://developer.adobe.com/firefly-services/docs/indesign-apis/getting_started/usage/
MAX_FILE_SIZE_GB = 1
MAX_SCRIPT_SIZE_MB = 5
MAX_ASSETS_PER_PAYLOAD = 99
RATE_LIMIT_SOFT = 250  # requests per minute (may slow down)
RATE_LIMIT_HARD = 350  # requests per minute (rejected beyond this)
OUTPUT_RETENTION_HOURS = 12  # Azure blob storage retention without presigned URL

# Initialize RAG system (lazy loading)
_rag_instance = None

def get_rag():
    """Get or create RAG instance."""
    global _rag_instance
    if _rag_instance is None:
        try:
            from api_rag import InDesignAPIRAG
            _rag_instance = InDesignAPIRAG()
        except Exception as e:
            print(f"RAG system not available: {e}")
            _rag_instance = False
    return _rag_instance if _rag_instance else None


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


async def _http_request(
    method: str, 
    url: str, 
    headers: dict, 
    json_body: dict | None = None,
    max_retries: int = 3
) -> dict | None:
    """
    Make HTTP request to Adobe API with automatic retry logic.
    
    Implements Adobe's recommended retry strategy:
    - Only retries 5xx server errors (not 4xx client errors)
    - Exponential backoff: waits 2^attempt seconds between retries  
    - Maximum 3 retry attempts
    
    Reference: https://developer.adobe.com/firefly-services/docs/indesign-apis/getting_started/usage/#api-retry
    """
    async with httpx.AsyncClient(timeout=120.0) as client:
        for attempt in range(max_retries):
            try:
                resp = await client.request(method=method, url=url, headers=headers, json=json_body)
                resp.raise_for_status()
                return resp.json() if resp.text else {}
            except httpx.HTTPStatusError as e:
                # Only retry 5xx errors (server-side issues)
                if 500 <= e.response.status_code < 600 and attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    await asyncio.sleep(wait_time)
                    continue
                return {"error": f"HTTP {e.response.status_code}: {e.response.text}"}
            except Exception as e:
                # Retry on network errors
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    continue
                return {"error": str(e)}


async def _poll_status(status_url: str, max_attempts: int = 60, delay: int = 5) -> dict:
    """Poll status URL until job completes or times out."""
    headers = _resolve_auth()
    
    for attempt in range(max_attempts):
        data = await _http_request("GET", status_url, headers)
        
        if not data or "error" in data:
            return {"status": "failed", "error": data.get("error", "Unknown error")}
        
        status = data.get("status", "unknown")
        
        if status in ["succeeded", "partial_success"]:
            return data
        elif status == "failed":
            return data
        
        # Still running, wait and try again
        await asyncio.sleep(delay)
    
    return {"status": "timeout", "message": "Job did not complete within timeout period"}


def _extract_filename_from_url(url: str) -> str:
    """Extract and sanitize filename from URL."""
    try:
        basename = os.path.basename(urlsplit(url).path)
        return _sanitize_filename(unquote(basename))
    except Exception:
        return "document.indd"


def _build_s3_client(
    aws_access_key_id: str | None,
    aws_secret_access_key: str | None,
    aws_region: str | None,
    endpoint_url: str | None,
):
    """Build S3 client with optional credential overrides."""
    resolved_key = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
    resolved_secret = aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")
    resolved_region = aws_region or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
    
    client_kwargs: dict[str, Any] = {
        "service_name": "s3",
        "region_name": resolved_region,
        "config": BotoConfig(signature_version="s3v4"),
    }
    if endpoint_url:
        client_kwargs["endpoint_url"] = endpoint_url
    if resolved_key and resolved_secret:
        client_kwargs["aws_access_key_id"] = resolved_key
        client_kwargs["aws_secret_access_key"] = resolved_secret
    return boto3.client(**client_kwargs)


# ============ Core MCP Tools ============

@mcp.tool
async def intelligent_indesign_api(
    prompt: str,
    source_url: str,
    additional_files: Optional[Dict[str, str]] = None,
    wait_for_completion: bool = True
) -> str:
    """
    🎯 INTELLIGENT API - Automatically determines and executes the right InDesign API based on your prompt.
    
    This is the main tool that uses RAG to understand your request and call the appropriate API.
    
    ⚠️ Rate Limits: 250 req/min (soft), 350 req/min (hard)
    📏 File Limit: 1GB maximum
    🔄 Retry: Automatic exponential backoff for 5xx errors
    
    For detailed guides, see resources:
    - indesign://guides/rendition - PDF/JPEG/PNG best practices
    - indesign://guides/data-merge - CSV merge workflows  
    - indesign://technical-limits - All limits and constraints
    
    Reference: https://developer.adobe.com/firefly-services/docs/indesign-apis/guides/
    
    Args:
        prompt: Natural language description of what you want to do
        source_url: Pre-signed URL to the main InDesign document (max 1GB, supported storage)
        additional_files: Optional dict of additional file URLs (e.g., {"csv": csv_url, "image1": img_url})
                         Keys are file type labels, filenames extracted from URLs automatically
        wait_for_completion: If True, waits for job to complete and returns final result (default: True)
    
    Returns:
        JSON with API response and results
    
    Examples:
        - "Convert this to PDF at high quality"
        - "Export pages 1-5 as JPEG at 300 DPI"
        - "Get all fonts and links from this document"
        - "Merge this template with data.csv"
        
        # Data merge with images (filenames extracted from URLs):
        intelligent_indesign_api(
            "Merge template with CSV data",
            template_url,
            additional_files={
                "csv": "https://storage/.../data.csv?...",
                "image1": "https://storage/.../face1.jpg?...",  # face1.jpg extracted from URL
                "image2": "https://storage/.../face2.jpg?..."   # face2.jpg extracted from URL
            }
        )
        # CRITICAL: Ensure image URLs contain correct filenames matching CSV references
    
    Note: For specific operations, dedicated tools may be more reliable than natural language.
    For image merge: URL paths must contain exact filenames that match CSV image references.
    """
    rag = get_rag()
    if not rag:
        return json.dumps({
            "error": "RAG system not available",
            "hint": "Run: python api_rag.py --build"
        })
    
    try:
        # Use RAG to generate the appropriate payload
        payload_info = rag.generate_payload(prompt)
        
        if "error" in payload_info:
            return json.dumps({
                "error": "Could not determine API from prompt",
                "prompt": prompt,
                "suggestion": "Try being more specific, e.g., 'create PDF', 'get document info', 'merge data'"
            }, indent=2)
        
        endpoint = payload_info["endpoint"]
        payload = payload_info["payload"]
        
        # Build fresh assets array with actual URLs (replace RAG template placeholders)
        new_assets = []
        
        # First asset is always the main source
        main_filename = _extract_filename_from_url(source_url)
        new_assets.append({
            "source": {
                "url": source_url,
                "storageType": _infer_storage_type(source_url)
            },
            "destination": main_filename
        })
        
        # Add additional files if provided (for data merge, etc.)
        csv_filename = None
        if additional_files:
            for file_type, file_url in additional_files.items():
                file_dest = _extract_filename_from_url(file_url)
                new_assets.append({
                    "source": {
                        "url": file_url,
                        "storageType": _infer_storage_type(file_url)
                    },
                    "destination": file_dest
                })
                
                # Track CSV filename for dataSource param
                if file_type == "csv":
                    csv_filename = file_dest
        
        # Replace assets in payload
        payload["assets"] = new_assets
        
        # Update params with correct filenames
        if "params" in payload:
            # Update target document references
            if "targetDocument" in payload["params"]:
                payload["params"]["targetDocument"] = main_filename
            if "targetDocuments" in payload["params"]:
                payload["params"]["targetDocuments"] = [main_filename]
            
            # Update dataSource for data merge
            if csv_filename and "dataSource" in payload["params"]:
                payload["params"]["dataSource"] = csv_filename
        
        # Determine the API URL
        method, path = endpoint.split(" ", 1)
        api_url = f"{ADOBE_API_BASE}{path}"
        
        # Execute the API call
        headers = _resolve_auth()
        result = await _http_request(method, api_url, headers, payload)
        
        if not result or "error" in result:
            return json.dumps({
                "status": "failed",
                "error": result.get("error", "API call failed"),
                "endpoint": endpoint,
                "payload_used": payload
            }, indent=2)
        
        # Get status URL
        status_url = result.get("statusUrl") or result.get("status_url")
        
        if not status_url:
            # No status URL means immediate response
            return json.dumps({
                "status": "completed",
                "result": result,
                "endpoint": endpoint
            }, indent=2)
        
        # If wait_for_completion, poll until done
        if wait_for_completion:
            final_result = await _poll_status(status_url)
            return json.dumps({
                "status": final_result.get("status"),
                "result": final_result,
                "endpoint": endpoint,
                "prompt": prompt
            }, indent=2)
        else:
            return json.dumps({
                "status": "submitted",
                "status_url": status_url,
                "job_id": result.get("jobId"),
                "endpoint": endpoint,
                "message": "Job submitted. Use check_job_status to monitor progress."
            }, indent=2)
    
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "prompt": prompt
        }, indent=2)


@mcp.tool
async def check_job_status(status_url: str, wait_for_completion: bool = False) -> str:
    """
    Check the status of an InDesign API job.
    
    Args:
        status_url: The status URL returned from any InDesign API call
        wait_for_completion: If True, polls until job completes (default: False)
    
    Returns:
        JSON with current job status and results if complete
    """
    if wait_for_completion:
        result = await _poll_status(status_url)
    else:
        headers = _resolve_auth()
        result = await _http_request("GET", status_url, headers)
    
    return json.dumps(result, indent=2)


@mcp.tool
async def upload_file_and_presign(
    file_path: str,
    object_key: str | None = None,
    expires_in_seconds: int = 3600,
    aws_access_key_id: str | None = None,
    aws_secret_access_key: str | None = None,
    aws_region: str | None = None,
    endpoint_url: str | None = None,
) -> str:
    """
    Upload a local file to S3 and return a pre-signed GET URL for InDesign API.
    
    Use this to upload InDesign files, CSVs, fonts, or images before processing.
    
    📦 Supported Storage (per Adobe):
    - AWS S3: ✅ Input & Output
    - Dropbox: ✅ Input & Output  
    - Azure: ✅ Input & Output
    - Google Cloud: ✅ Input only
    - Box.com: ✅ Input only
    - AEM: ✅ Input only
    
    ⚠️ Limits:
    - Max file size: 1GB (automatically validated)
    - Domain whitelisting: Contact Adobe for custom domains
    
    Reference: https://developer.adobe.com/firefly-services/docs/indesign-apis/getting_started/usage/
    
    Args:
        file_path: Local path to upload
        object_key: Optional key; defaults to the file's basename
        expires_in_seconds: URL expiry in seconds (default 3600)
        aws_access_key_id: Optional override; falls back to env
        aws_secret_access_key: Optional override; falls back to env
        aws_region: Optional override; falls back to env (default us-east-1)
        endpoint_url: Optional custom S3 endpoint
    
    Returns:
        Pre-signed URL as string
    
    Example:
        url = upload_file_and_presign('/path/to/document.indd')
        result = intelligent_indesign_api('Convert to PDF', url)
    """
    import mimetypes
    
    key = object_key or os.path.basename(file_path)
    bucket = os.getenv("S3_BUCKET") or os.getenv("AWS_S3_BUCKET")
    
    if not bucket:
        return json.dumps({
            "error": "S3_BUCKET not configured",
            "hint": "Set S3_BUCKET or AWS_S3_BUCKET environment variable"
        })
    
    if not os.path.exists(file_path):
        return json.dumps({
            "error": f"File not found: {file_path}"
        })
    
    # Validate file size (Adobe limit: 1GB)
    file_size_bytes = os.path.getsize(file_path)
    file_size_gb = file_size_bytes / (1024 ** 3)
    
    if file_size_gb > MAX_FILE_SIZE_GB:
        return json.dumps({
            "error": f"File too large: {file_size_gb:.2f}GB",
            "limit": f"{MAX_FILE_SIZE_GB}GB (Adobe InDesign API limit)",
            "file": file_path,
            "reference": "https://developer.adobe.com/firefly-services/docs/indesign-apis/getting_started/usage/#file-size"
        })
    
    s3 = _build_s3_client(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_region=aws_region,
        endpoint_url=endpoint_url,
    )
    
    content_type, _ = mimetypes.guess_type(file_path)
    extra_args = {"ContentType": content_type} if content_type else None
    
    try:
        if extra_args:
            s3.upload_file(file_path, bucket, key, ExtraArgs=extra_args)
        else:
            s3.upload_file(file_path, bucket, key)
        
        url = s3.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expires_in_seconds,
        )
        
        return url
        
    except Exception as exc:
        return json.dumps({
            "error": f"Upload/presign failed: {exc}"
        })


# ============ Direct API Tools (for advanced users) ============

@mcp.tool
async def create_rendition_direct(
    source_url: str,
    output_format: str,
    page_range: str = "All",
    resolution: int = 72,
    quality: str = "medium",
    wait_for_completion: bool = True
) -> str:
    """
    Create PDF, JPEG, or PNG rendition (direct API call).
    
    📐 Resolution Guidelines (from Adobe):
    - 72 DPI: Web graphics, screen viewing
    - 150 DPI: Proofing, draft prints
    - 300 DPI: Print-ready, professional output
    
    🎨 Quality Guidelines:
    - 'low'/'medium': Web distribution, quick proofs
    - 'high'/'maximum': Print-ready, final production
    
    For detailed best practices, see: indesign://guides/rendition
    Reference: https://developer.adobe.com/firefly-services/docs/indesign-apis/guides/
    
    Args:
        source_url: Pre-signed URL to InDesign file
        output_format: 'pdf', 'jpeg', or 'png'
        page_range: Pages to export - 'All', '1-5', '1,3,5' (default: 'All')
        resolution: DPI - 72 (web), 150 (proof), 300 (print) (default: 72)
        quality: 'low', 'medium', 'high', 'maximum' (default: 'medium')
        wait_for_completion: Wait for job to finish (default: True)
    
    Example:
        # High-quality PDF for print
        create_rendition_direct(source_url, 'pdf', quality='high')
        
        # Web-optimized JPEG
        create_rendition_direct(source_url, 'jpeg', resolution=72, quality='medium')
    
    Note: For most users, intelligent_indesign_api is easier with natural language.
    """
    format_map = {"pdf": "application/pdf", "jpeg": "image/jpeg", "jpg": "image/jpeg", "png": "image/png"}
    output_media_type = format_map.get(output_format.lower(), output_format)
    
    headers = _resolve_auth()
    url = f"{ADOBE_API_BASE}/{ADOBE_API_VERSION}/create-rendition"
    
    target_doc = _extract_filename_from_url(source_url)
    
    payload = {
        "assets": [{
            "source": {"url": source_url, "storageType": _infer_storage_type(source_url)},
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
    
    result = await _http_request("POST", url, headers, payload)
    
    if wait_for_completion and result and "statusUrl" in result:
        final_result = await _poll_status(result["statusUrl"])
        return json.dumps(final_result, indent=2)
    
    return json.dumps(result, indent=2)


@mcp.tool
async def get_document_info_direct(
    source_url: str,
    get_fonts: bool = True,
    get_links: bool = True,
    get_pages: bool = True,
    wait_for_completion: bool = True
) -> str:
    """
    Get document information (direct API call).
    
    For most users, use intelligent_indesign_api instead.
    
    Args:
        source_url: Pre-signed URL to InDesign file
        get_fonts: Include fonts (default: True)
        get_links: Include links (default: True)
        get_pages: Include pages (default: True)
        wait_for_completion: Wait for job to finish (default: True)
    """
    headers = _resolve_auth()
    url = f"{ADOBE_API_BASE}/{ADOBE_API_VERSION}/document-info"
    
    target_doc = _extract_filename_from_url(source_url)
    
    payload = {
        "assets": [{
            "source": {"url": source_url, "storageType": _infer_storage_type(source_url)},
            "destination": target_doc
        }],
        "params": {
            "targetDocument": target_doc,
            "pageInfo": {"enabled": get_pages},
            "linkInfo": {"enabled": get_links},
            "fontInfo": {"enabled": get_fonts}
        }
    }
    
    result = await _http_request("POST", url, headers, payload)
    
    if wait_for_completion and result and "statusUrl" in result:
        final_result = await _poll_status(result["statusUrl"])
        return json.dumps(final_result, indent=2)
    
    return json.dumps(result, indent=2)


@mcp.tool
async def get_data_merge_tags(
    source_url: str,
    csv_url: str | None = None,
    filter_types: list[str] | None = None,
    wait_for_completion: bool = True
) -> str:
    """
    Extract data merge tags/placeholders from an InDesign template.
    
    Use this to discover what merge fields are available in a template before merging.
    
    Args:
        source_url: Pre-signed URL to InDesign template file
        csv_url: Optional CSV file URL to match tags against
        filter_types: Filter by tag type: ['text', 'image', 'qr'] or None for all (default: None)
        wait_for_completion: Wait for job to finish (default: True)
    
    Returns:
        JSON with list of data merge tags found in the template
    
    Example:
        get_data_merge_tags(template_url)
        get_data_merge_tags(template_url, csv_url, filter_types=['text', 'image'])
    """
    headers = _resolve_auth()
    url = f"{ADOBE_API_BASE}/{ADOBE_API_VERSION}/merge-data-tags"
    
    target_doc = _extract_filename_from_url(source_url)
    
    # Build assets array
    assets = [{
        "source": {"url": source_url, "storageType": _infer_storage_type(source_url)},
        "destination": target_doc
    }]
    
    # Add CSV if provided
    csv_dest = None
    if csv_url:
        csv_dest = _extract_filename_from_url(csv_url)
        assets.append({
            "source": {"url": csv_url, "storageType": _infer_storage_type(csv_url)},
            "destination": csv_dest
        })
    
    # Build params
    params = {
        "targetDocument": target_doc
    }
    
    if csv_dest:
        params["dataSource"] = csv_dest
    
    if filter_types:
        params["filter"] = filter_types
    else:
        params["filter"] = ["all"]
    
    payload = {
        "assets": assets,
        "params": params
    }
    
    result = await _http_request("POST", url, headers, payload)
    
    if wait_for_completion and result and "statusUrl" in result:
        final_result = await _poll_status(result["statusUrl"])
        return json.dumps(final_result, indent=2)
    
    return json.dumps(result, indent=2)


@mcp.tool
async def data_merge(
    template_url: str,
    csv_url: str,
    output_format: str = "pdf",
    record_range: str = "All",
    wait_for_completion: bool = True
) -> str:
    """
    Merge CSV data with InDesign template to create personalized documents.
    
    Creates multiple document variations based on CSV rows (one output per row).
    
    💡 CSV ENCODING (per Adobe):
    - UTF-16BE required for multi-byte characters (Asian languages, special symbols)
    - Plain English CSV works without special encoding
    
    🖼️ IMAGES IN DATA MERGE:
    - CSV can reference image filenames (e.g., Photo column with "face1.jpg")
    - Template uses @<<Photo>>@ syntax for image placeholders
    - Images must be provided as additional assets with matching filenames
    - NOTE: This tool doesn't support images - use intelligent_indesign_api instead
    
    📝 Best Practices:
    - Test with record_range='1-3' before processing all records
    - Verify merge tags match CSV headers exactly (case-sensitive)
    - Use descriptive field names (e.g., 'CustomerName' not 'field1')
    
    📦 Path Management:
    - Outputs stored in Adobe Azure blob (12h) if no custom output path specified
    - For production: Specify output destinations in assets array
    - Use placeholders: {record}, {CustomerName}, etc.
    
    For detailed guides:
    - indesign://guides/data-merge - Complete workflow including image merge
    - indesign://guides/asset-paths - Input/output path details
    
    Reference: https://developer.adobe.com/firefly-services/docs/indesign-apis/guides/
    
    Args:
        template_url: Pre-signed URL to InDesign template with merge fields (<<field_name>>)
        csv_url: Pre-signed URL to CSV file with matching column headers
        output_format: Output format - 'pdf', 'jpeg', 'png', or 'indesign' (default: 'pdf')
        record_range: Records to process - 'All', '1-10', '1,3,5', etc. (default: 'All')
        wait_for_completion: Wait for job to finish (default: True)
    
    Returns:
        JSON with merged document URLs
    
    Example:
        # Test first 3 records
        data_merge(template_url, csv_url, output_format="pdf", record_range="1-3")
        
        # Process all records
        data_merge(template_url, csv_url, output_format="pdf", record_range="All")
    
    Note: For merges with images, use intelligent_indesign_api with additional_files parameter.
    """
    format_map = {
        "pdf": "application/pdf",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "indesign": "application/x-indesign"
    }
    output_media_type = format_map.get(output_format.lower(), output_format)
    
    headers = _resolve_auth()
    url = f"{ADOBE_API_BASE}/{ADOBE_API_VERSION}/merge-data"
    
    template_dest = _extract_filename_from_url(template_url)
    csv_dest = _extract_filename_from_url(csv_url)
    
    payload = {
        "assets": [
            {
                "source": {"url": template_url, "storageType": _infer_storage_type(template_url)},
                "destination": template_dest
            },
            {
                "source": {"url": csv_url, "storageType": _infer_storage_type(csv_url)},
                "destination": csv_dest
            }
        ],
        "params": {
            "targetDocument": template_dest,
            "dataSource": csv_dest,
            "outputMediaType": output_media_type,
            "recordRange": record_range
        }
    }
    
    result = await _http_request("POST", url, headers, payload)
    
    if wait_for_completion and result and "statusUrl" in result:
        final_result = await _poll_status(result["statusUrl"])
        return json.dumps(final_result, indent=2)
    
    return json.dumps(result, indent=2)


# ============ RAG Helper Tools ============

@mcp.tool
async def query_documentation(query: str, num_results: int = 3) -> str:
    """
    Search InDesign API documentation using natural language.
    
    Args:
        query: Your question (e.g., "What quality options are available?")
        num_results: Number of results (default: 3)
    """
    rag = get_rag()
    if not rag:
        return json.dumps({"error": "RAG not available"})
    
    results = rag.query(query, n_results=num_results)
    formatted = [{
        "rank": i+1,
        "relevance": 1 - r.get("distance", 1),
        "content": r["content"][:400] + "..."
    } for i, r in enumerate(results)]
    
    return json.dumps({"query": query, "results": formatted}, indent=2)


# ============ Resources ============

@mcp.resource("indesign://status")
def system_status() -> str:
    """Get system status and configuration."""
    rag = get_rag()
    
    return json.dumps({
        "adobe_api": {
            "base_url": ADOBE_API_BASE,
            "version": ADOBE_API_VERSION,
            "api_key_configured": bool(os.getenv("ADOBE_API_KEY")),
            "access_token_configured": bool(os.getenv("ADOBE_ACCESS_TOKEN"))
        },
        "s3": {
            "bucket_configured": bool(os.getenv("S3_BUCKET") or os.getenv("AWS_S3_BUCKET")),
            "region": os.getenv("AWS_REGION") or "us-east-1"
        },
        "rag": {
            "available": rag is not None,
            "database_exists": os.path.exists(".chromadb")
        },
        "supported_apis": [
            "Rendition (PDF, JPEG, PNG)",
            "Document Info (fonts, links, pages)",
            "Data Merge (template + CSV)",
            "Data Merge Tags (extract tags)",
            "Remap Links (file to AEM)"
        ]
    }, indent=2)


@mcp.resource("indesign://technical-limits")
def technical_limits() -> str:
    """
    Adobe InDesign API technical limits and usage guidelines.
    
    Reference: https://developer.adobe.com/firefly-services/docs/indesign-apis/getting_started/usage/
    """
    return json.dumps({
        "rate_limits": {
            "soft_limit": f"{RATE_LIMIT_SOFT} requests/minute",
            "hard_limit": f"{RATE_LIMIT_HARD} requests/minute",
            "soft_behavior": "May experience slower responses",
            "hard_behavior": "Requests rejected",
            "recommendation": "Implement rate limiting in your application"
        },
        "file_limits": {
            "max_file_size": f"{MAX_FILE_SIZE_GB}GB",
            "max_script_size": f"{MAX_SCRIPT_SIZE_MB}MB (Custom Scripts API)",
            "max_assets_per_payload": MAX_ASSETS_PER_PAYLOAD,
            "note": "Input + output assets combined"
        },
        "storage_support": {
            "input": ["AWS S3", "Dropbox", "Azure", "AEM", "Google Cloud", "Box.com"],
            "output": ["AWS S3", "Dropbox", "Azure"],
            "domain_whitelisting": "Contact Adobe to whitelist custom domains"
        },
        "data_retention": {
            "input_assets": "Deleted after processing completes",
            "output_with_presigned_url": "Uploaded to your storage (permanent)",
            "output_without_presigned_url": f"{OUTPUT_RETENTION_HOURS} hours in Adobe Azure blob",
            "metadata": "Stored in database (file names, sizes, URLs)",
            "scripts": "Stored permanently (user can delete)"
        },
        "retry_strategy": {
            "when": "Only retry 5xx server errors",
            "dont_retry": "4xx client errors (bad request, auth issues)",
            "strategy": "Exponential backoff",
            "max_attempts": 3,
            "implementation": "Built into this MCP server automatically"
        }
    }, indent=2)


@mcp.resource("indesign://guides/rendition")
def rendition_guide() -> str:
    """
    Best practices for working with the Rendition API.
    
    Reference: https://developer.adobe.com/firefly-services/docs/indesign-apis/guides/
    """
    return json.dumps({
        "overview": "Create PDF, JPEG, PNG renditions from InDesign documents",
        "output_formats": {
            "pdf": {
                "media_type": "application/pdf",
                "quality_presets": ["low", "medium", "high", "maximum"],
                "use_cases": ["print-ready documents", "digital distribution", "proofing"],
                "best_practice": "Use 'high' or 'maximum' for print, 'medium' for web"
            },
            "jpeg": {
                "media_type": "image/jpeg",
                "resolution_options": [72, 150, 300],
                "use_cases": ["web graphics", "social media", "presentations"],
                "best_practice": "72 DPI for web, 300 DPI for print"
            },
            "png": {
                "media_type": "image/png",
                "supports_transparency": True,
                "use_cases": ["logos with transparency", "web graphics", "UI elements"],
                "best_practice": "Use for images requiring transparency"
            }
        },
        "parameters": {
            "pageRange": {
                "options": ["All", "1-5", "1,3,5", "1-3,7-9"],
                "default": "All",
                "tip": "Export specific pages to reduce processing time"
            },
            "resolution": {
                "values": [72, 150, 300, 600],
                "default": 72,
                "recommendation": "72 for web, 300 for print, 150 for proofs"
            },
            "quality": {
                "values": ["low", "medium", "high", "maximum"],
                "default": "medium",
                "trade_off": "Higher quality = larger file size + longer processing"
            }
        },
        "tips": [
            "Always provide presigned URLs for input documents",
            "Wait for completion or poll status for long jobs",
            "Use appropriate quality/resolution for your use case",
            "Export specific page ranges for faster processing",
            "Monitor rate limits (250 soft / 350 hard per minute)"
        ],
        "tools_available": [
            "intelligent_indesign_api - Natural language (e.g., 'Convert to PDF at high quality')",
            "create_rendition_direct - Explicit parameters for full control"
        ]
    }, indent=2)


@mcp.resource("indesign://guides/data-merge")
def data_merge_guide() -> str:
    """
    Best practices for working with the Data Merge API.
    
    Reference: https://developer.adobe.com/firefly-services/docs/indesign-apis/guides/
    """
    return json.dumps({
        "overview": "Create personalized documents by merging CSV data with InDesign templates",
        "workflow": {
            "step_1": "Create InDesign template with data merge placeholders (<<field_name>>)",
            "step_2": "Prepare CSV file with matching column headers",
            "step_3": "Upload both files and get presigned URLs",
            "step_4": "Optional: Extract merge tags to verify field names",
            "step_5": "Perform data merge to generate personalized outputs"
        },
        "input_path_management": {
            "overview": "How InDesign API handles input files in the assets array",
            "asset_structure": {
                "source": {
                    "url": "Pre-signed URL to the file (S3, Dropbox, Azure, etc.)",
                    "storageType": "AWS, Dropbox, Azure, external - auto-detected from URL"
                },
                "destination": {
                    "purpose": "Internal filename used by InDesign during processing",
                    "example": "template.indd or data.csv",
                    "important": "Must match targetDocument and dataSource params"
                }
            },
            "example_payload": {
                "assets": [
                    {
                        "source": {"url": "https://bucket.s3.amazonaws.com/file.indd?...", "storageType": "AWS"},
                        "destination": "template.indd"
                    },
                    {
                        "source": {"url": "https://bucket.s3.amazonaws.com/data.csv?...", "storageType": "AWS"},
                        "destination": "data.csv"
                    }
                ],
                "params": {
                    "targetDocument": "template.indd",
                    "dataSource": "data.csv"
                }
            },
            "key_points": [
                "destination field is the internal reference name (not an output path)",
                "targetDocument must exactly match the template asset destination",
                "dataSource must exactly match the CSV asset destination",
                "File extensions should match actual file type",
                "Destination names are case-sensitive"
            ]
        },
        "output_path_management": {
            "overview": "How InDesign API handles output file storage and naming",
            "default_behavior": {
                "no_output_specified": "Files uploaded to Adobe's Azure blob storage",
                "retention": "12 hours only",
                "access": "Presigned URLs returned in response",
                "naming": "Auto-generated based on template name and record number"
            },
            "output_naming_pattern": {
                "single_output": "template_name.pdf",
                "multiple_records": [
                    "template_name_1.pdf",
                    "template_name_2.pdf",
                    "template_name_3.pdf"
                ],
                "note": "Output names derived from template asset destination name"
            },
            "custom_output_paths": {
                "how_to": "Add output assets to the assets array with destination URLs",
                "example": {
                    "assets": [
                        {"source": {...}, "destination": "template.indd"},
                        {"source": {...}, "destination": "data.csv"},
                        {
                            "destination": {
                                "url": "https://your-bucket.s3.amazonaws.com/output/result_{record}.pdf",
                                "storageType": "AWS"
                            }
                        }
                    ],
                    "note": "Use {record} placeholder for record number in output names"
                },
                "placeholders": {
                    "{record}": "Record number (1, 2, 3, ...)",
                    "{field_name}": "Value from CSV field",
                    "example": "output_{CustomerName}_{record}.pdf"
                },
                "requirements": [
                    "Output storage must support write operations (AWS, Dropbox, Azure)",
                    "Presigned URLs must have write permissions",
                    "URL must be valid for the duration of the merge job",
                    "Path placeholders replaced per record"
                ]
            },
            "storage_options": {
                "azure_default": {
                    "when": "No output destination specified",
                    "retention": "12 hours",
                    "access": "Presigned URLs in API response",
                    "cost": "Free (included in API)"
                },
                "your_storage": {
                    "when": "Output destination URLs provided",
                    "retention": "Permanent (your control)",
                    "access": "Direct access to your storage",
                    "cost": "Your storage costs"
                }
            },
            "best_practices": [
                "For production: Always specify output destinations to your storage",
                "Use meaningful output names with record identifiers",
                "Ensure output URLs have sufficient expiry time",
                "Use path placeholders to organize outputs by customer/project",
                "Test output paths with record_range='1-3' first"
            ]
        },
        "csv_encoding": {
            "supported": "UTF-16BE",
            "when_required": "Multi-byte characters (Asian languages, special symbols)",
            "english_note": "Plain English CSV works without UTF-16BE",
            "how_to_encode": "Save CSV with UTF-16BE encoding from Excel or text editor"
        },
        "output_formats": {
            "pdf": "One PDF per CSV row (most common)",
            "jpeg": "One JPEG per CSV row",
            "png": "One PNG per CSV row",
            "indesign": "One .indd file per CSV row (for further editing)"
        },
        "record_range": {
            "syntax": ["All", "1-10", "1,3,5", "1-3,7-9"],
            "examples": {
                "All": "Process every row in CSV",
                "1-5": "Process rows 1 through 5",
                "1,3,5": "Process rows 1, 3, and 5 only",
                "1-3,7-9": "Process rows 1-3 and 7-9"
            },
            "tip": "Use specific ranges for testing before processing all records"
        },
        "image_merge": {
            "overview": "Include images in data merge by referencing them in CSV and providing as assets",
            "how_it_works": {
                "step_1": "InDesign template has image placeholder: @<<ImageField>>@",
                "step_2": "CSV has column 'ImageField' with image filenames: face1.jpg, face2.jpg",
                "step_3": "Provide each image as an asset with destination matching CSV filename",
                "step_4": "InDesign downloads images and merges them into template per record"
            },
            "csv_structure": {
                "headers": ["Name", "Email", "Photo"],
                "row_1": ["John Doe", "john@example.com", "face1.jpg"],
                "row_2": ["Jane Smith", "jane@example.com", "face2.jpg"],
                "note": "Photo column contains exact filenames matching asset destinations"
            },
            "template_setup": {
                "text_placeholder": "<<Name>> - <<Email>>",
                "image_placeholder": "@<<Photo>>@",
                "important": "Use @<<FieldName>>@ syntax for image placeholders in InDesign"
            },
            "payload_structure": {
                "assets": [
                    {
                        "source": {"url": "https://.../Template.indd?...", "storageType": "Azure"},
                        "destination": "Template.indd"
                    },
                    {
                        "source": {"url": "https://.../data.csv?...", "storageType": "Azure"},
                        "destination": "data.csv"
                    },
                    {
                        "source": {"url": "https://.../face1.jpg?...", "storageType": "Azure"},
                        "destination": "face1.jpg"
                    },
                    {
                        "source": {"url": "https://.../face2.jpg?...", "storageType": "Azure"},
                        "destination": "face2.jpg"
                    }
                ],
                "params": {
                    "targetDocument": "Template.indd",
                    "dataSource": "data.csv",
                    "outputMediaType": "application/pdf"
                }
            },
            "critical_rules": [
                "Image destination MUST exactly match filename in CSV (case-sensitive)",
                "Include ALL images referenced in ANY row of the CSV",
                "Image formats: JPG, PNG, TIFF, PSD, AI, PDF",
                "Images downloaded once and cached for entire merge operation",
                "Missing images will cause merge to fail for affected records",
                "Presigned URLs for images must be valid for entire job duration"
            ],
            "workflow": {
                "1_prepare_template": "Create InDesign template with @<<Photo>>@ placeholder",
                "2_prepare_csv": "CSV with 'Photo' column containing 'face1.jpg', 'face2.jpg', etc.",
                "3_upload_files": "Upload template, CSV, and ALL referenced images",
                "4_get_presigned_urls": "Generate presigned URLs for each file",
                "5_build_assets": "Add template + CSV + each image to assets array",
                "6_match_names": "Ensure image destinations match CSV filenames exactly",
                "7_execute": "Call data merge API with complete payload"
            },
            "real_example": {
                "description": "Merge employee badges with photos",
                "csv": {
                    "headers": ["EmployeeID", "Name", "Department", "Photo"],
                    "data": [
                        ["1001", "John Doe", "Engineering", "john_photo.jpg"],
                        ["1002", "Jane Smith", "Marketing", "jane_photo.jpg"]
                    ]
                },
                "assets": [
                    "Template.indd (badge template with @<<Photo>>@)",
                    "employees.csv (data file)",
                    "john_photo.jpg (matches CSV row 1)",
                    "jane_photo.jpg (matches CSV row 2)"
                ],
                "result": [
                    "badge_1.pdf (John's badge with john_photo.jpg)",
                    "badge_2.pdf (Jane's badge with jane_photo.jpg)"
                ]
            }
        },
        "complete_example": {
            "scenario": "Merge 100 customer letters with custom output paths",
            "payload": {
                "assets": [
                    {
                        "source": {"url": "https://bucket.s3.amazonaws.com/letter.indd?...", "storageType": "AWS"},
                        "destination": "letter_template.indd"
                    },
                    {
                        "source": {"url": "https://bucket.s3.amazonaws.com/customers.csv?...", "storageType": "AWS"},
                        "destination": "customer_data.csv"
                    },
                    {
                        "destination": {
                            "url": "https://bucket.s3.amazonaws.com/output/customer_{CustomerID}_letter_{record}.pdf",
                            "storageType": "AWS"
                        }
                    }
                ],
                "params": {
                    "targetDocument": "letter_template.indd",
                    "dataSource": "customer_data.csv",
                    "outputMediaType": "application/pdf",
                    "recordRange": "All"
                }
            },
            "result": [
                "customer_1001_letter_1.pdf",
                "customer_1002_letter_2.pdf",
                "customer_1003_letter_3.pdf"
            ]
        },
        "common_errors": {
            "missing_fields": "CSV missing columns that template expects",
            "encoding_issues": "Special characters not displaying - use UTF-16BE",
            "placeholder_mismatch": "Field names in template don't match CSV headers",
            "destination_mismatch": "targetDocument doesn't match template asset destination",
            "invalid_output_url": "Output URL doesn't have write permissions or is expired",
            "storage_type_error": "Trying to write to read-only storage (GCP, Box, AEM)"
        },
        "tools_available": [
            "get_data_merge_tags - Extract placeholders from template (verify before merge)",
            "data_merge - Perform the actual merge operation",
            "intelligent_indesign_api - Natural language merge requests"
        ]
    }, indent=2)


@mcp.resource("indesign://guides/asset-paths")
def asset_paths_guide() -> str:
    """
    Comprehensive guide to input and output path management across all InDesign APIs.
    
    Reference: https://developer.adobe.com/firefly-services/docs/indesign-apis/guides/
    """
    return json.dumps({
        "overview": "Understanding how InDesign APIs handle file paths for inputs and outputs",
        "input_asset_structure": {
            "purpose": "Tell InDesign where to download input files from",
            "required_fields": {
                "source": {
                    "url": "Pre-signed URL to download the file",
                    "storageType": "Storage provider type (AWS, Dropbox, Azure, external)"
                },
                "destination": "Internal filename InDesign uses during processing"
            },
            "example": {
                "source": {
                    "url": "https://my-bucket.s3.amazonaws.com/designs/newsletter.indd?X-Amz-...",
                    "storageType": "AWS"
                },
                "destination": "newsletter.indd"
            },
            "key_concepts": {
                "url": "Must be accessible presigned URL with read permissions",
                "storageType": "Auto-detected from URL, but can be specified explicitly",
                "destination": "Internal reference name - NOT an output path",
                "filename_matching": "Extensions should match actual file type (.indd, .csv, .pdf)"
            }
        },
        "destination_field_explained": {
            "what_it_is": "Internal filename used during InDesign processing",
            "what_it_is_NOT": "NOT the output filename or path",
            "how_its_used": [
                "Referenced in params.targetDocument or params.targetDocuments",
                "Referenced in params.dataSource for CSV files",
                "Must be consistent between assets and params",
                "Case-sensitive matching required"
            ],
            "example_flow": {
                "step_1": "Asset has destination='template.indd'",
                "step_2": "InDesign downloads file and stores it internally as 'template.indd'",
                "step_3": "params.targetDocument='template.indd' references this internal file",
                "step_4": "Processing happens on 'template.indd'",
                "step_5": "Output generated (separate from destination field)"
            }
        },
        "output_management": {
            "option_1_default": {
                "description": "No output asset specified - Adobe stores outputs",
                "behavior": {
                    "storage": "Adobe's Azure blob storage",
                    "retention": "12 hours only",
                    "access": "Presigned URLs returned in API response",
                    "naming": "Auto-generated from input filename",
                    "cost": "Free (included)"
                },
                "when_to_use": "Quick tests, temporary processing, prototyping",
                "example_response": {
                    "outputs": [
                        {"href": "https://firefly-api-prod.azure.net/temp/output_1.pdf?..."},
                        {"href": "https://firefly-api-prod.azure.net/temp/output_2.pdf?..."}
                    ]
                }
            },
            "option_2_custom": {
                "description": "Specify output destinations - files uploaded to your storage",
                "behavior": {
                    "storage": "Your AWS/Dropbox/Azure",
                    "retention": "Permanent (your control)",
                    "access": "Direct access from your storage",
                    "naming": "Custom with placeholders",
                    "cost": "Your storage costs"
                },
                "when_to_use": "Production workflows, permanent storage, organized outputs",
                "how_to_specify": {
                    "add_output_asset": {
                        "destination": {
                            "url": "https://your-bucket.s3.amazonaws.com/outputs/file_{record}.pdf",
                            "storageType": "AWS"
                        }
                    },
                    "note": "This is a destination object, not a string like input assets"
                },
                "output_placeholders": {
                    "{record}": "Record number for data merge (1, 2, 3...)",
                    "{field_name}": "CSV field value (e.g., {CustomerName}, {OrderID})",
                    "{timestamp}": "Processing timestamp",
                    "examples": [
                        "customer_{CustomerID}_invoice_{record}.pdf",
                        "output_{record}_{Date}.pdf",
                        "letters/{Region}/customer_{CustomerName}.pdf"
                    ]
                },
                "requirements": [
                    "URL must have write permissions",
                    "Storage must support output (AWS/Dropbox/Azure only)",
                    "URL must be valid for job duration",
                    "Path can include directories (will be created)"
                ]
            }
        },
        "common_patterns": {
            "single_file_processing": {
                "scenario": "Convert one InDesign file to PDF",
                "assets": [
                    {"source": {"url": "...", "storageType": "AWS"}, "destination": "doc.indd"}
                ],
                "output": "Auto-generated, returns presigned URL"
            },
            "data_merge_default_output": {
                "scenario": "Merge with temporary outputs",
                "assets": [
                    {"source": {"url": "...", "storageType": "AWS"}, "destination": "template.indd"},
                    {"source": {"url": "...", "storageType": "AWS"}, "destination": "data.csv"}
                ],
                "params": {"targetDocument": "template.indd", "dataSource": "data.csv"},
                "output": "Multiple files in Azure blob, 12h retention"
            },
            "data_merge_custom_output": {
                "scenario": "Merge with permanent organized outputs",
                "assets": [
                    {"source": {"url": "...", "storageType": "AWS"}, "destination": "template.indd"},
                    {"source": {"url": "...", "storageType": "AWS"}, "destination": "data.csv"},
                    {"destination": {"url": "s3://bucket/customers/{CustomerID}_{record}.pdf", "storageType": "AWS"}}
                ],
                "params": {"targetDocument": "template.indd", "dataSource": "data.csv"},
                "output": "Files uploaded directly to your S3 paths"
            },
            "data_merge_with_images": {
                "scenario": "Merge with images referenced in CSV",
                "description": "Images are downloaded by InDesign and merged into template per record",
                "csv_content": {
                    "headers": ["Name", "Email", "Photo"],
                    "rows": [
                        ["John Doe", "john@example.com", "face1.jpg"],
                        ["Jane Smith", "jane@example.com", "face2.jpg"]
                    ]
                },
                "assets": [
                    {"source": {"url": "https://.../Template.indd?...", "storageType": "Azure"}, "destination": "Template.indd"},
                    {"source": {"url": "https://.../data.csv?...", "storageType": "Azure"}, "destination": "data.csv"},
                    {"source": {"url": "https://.../face1.jpg?...", "storageType": "Azure"}, "destination": "face1.jpg"},
                    {"source": {"url": "https://.../face2.jpg?...", "storageType": "Azure"}, "destination": "face2.jpg"}
                ],
                "params": {"targetDocument": "Template.indd", "dataSource": "data.csv"},
                "critical": [
                    "Image destination MUST match CSV filename exactly (case-sensitive)",
                    "Template uses @<<Photo>>@ for image placeholders",
                    "Include ALL images referenced in CSV",
                    "Images cached and reused across records"
                ],
                "output": "Each record merged with its specified image"
            }
        },
        "storage_compatibility": {
            "input_support": {
                "AWS_S3": True,
                "Dropbox": True,
                "Azure": True,
                "Google_Cloud": True,
                "Box": True,
                "AEM": True
            },
            "output_support": {
                "AWS_S3": True,
                "Dropbox": True,
                "Azure": True,
                "Google_Cloud": False,
                "Box": False,
                "AEM": False
            },
            "note": "GCP, Box, and AEM are read-only for InDesign APIs"
        },
        "debugging_path_issues": {
            "error_destination_mismatch": {
                "symptom": "File not found or targetDocument error",
                "cause": "params.targetDocument doesn't match asset destination",
                "fix": "Ensure exact case-sensitive match: asset.destination === params.targetDocument"
            },
            "error_output_failed": {
                "symptom": "Output upload failed",
                "cause": "Output URL lacks write permissions or storage type unsupported",
                "fix": "Check presigned URL has PutObject permission, verify storage type is AWS/Dropbox/Azure"
            },
            "error_file_not_downloaded": {
                "symptom": "Could not download source file",
                "cause": "Input URL expired or inaccessible",
                "fix": "Regenerate presigned URL with longer expiry time"
            }
        },
        "best_practices": [
            "Use descriptive destination names that indicate file purpose",
            "Always match asset destination with params references exactly",
            "For production, specify custom output paths to your storage",
            "Use output path placeholders to organize files logically",
            "Set presigned URL expiry > expected job duration",
            "Test paths with small record ranges first",
            "Keep destination filenames simple (avoid special characters)",
            "Use folder structures in output URLs for organization"
        ]
    }, indent=2)


@mcp.resource("indesign://guides/logging")
def logging_guide() -> str:
    """
    Logging best practices for InDesign API operations.
    
    Reference: https://developer.adobe.com/firefly-services/docs/indesign-apis/guides/logging/
    """
    return json.dumps({
        "overview": "Essential logging techniques for debugging and monitoring",
        "what_to_log": {
            "api_requests": {
                "endpoint": "Which API endpoint was called",
                "payload": "Request payload (sanitize sensitive data)",
                "timestamp": "When the request was made",
                "user_context": "Who initiated the request"
            },
            "api_responses": {
                "status_code": "HTTP status code",
                "job_id": "Job ID for tracking",
                "status_url": "URL for polling job status",
                "errors": "Error messages and codes",
                "execution_time": "How long the request took"
            },
            "file_operations": {
                "uploads": "Files uploaded, sizes, URLs generated",
                "downloads": "Output files retrieved",
                "validation": "File size checks, format validation"
            }
        },
        "log_levels": {
            "ERROR": "Failed API calls, validation errors, exceptions",
            "WARN": "Rate limit approaching, large files, retries",
            "INFO": "Successful operations, job submissions, completions",
            "DEBUG": "Detailed payloads, intermediate steps, polling attempts"
        },
        "implementation": {
            "python_example": "Use Python's logging module with structured logs",
            "log_format": "Include timestamp, level, message, context (user, job_id)",
            "storage": "Centralized logging system (CloudWatch, Datadog, Splunk)",
            "retention": "Keep logs for 30-90 days for troubleshooting"
        },
        "this_server": {
            "automatic_logging": "Built-in retry logic logs attempts",
            "error_tracking": "All errors returned in JSON with context",
            "job_tracking": "Status URLs provided for long-running jobs"
        },
        "best_practices": [
            "Log all API calls with job IDs for traceability",
            "Include user context for debugging multi-user systems",
            "Monitor rate limit usage to avoid hitting hard limits",
            "Log file sizes to catch oversized uploads early",
            "Track processing times to identify performance issues",
            "Sanitize sensitive data (credentials, PII) from logs"
        ]
    }, indent=2)


@mcp.resource("indesign://examples")
def get_examples() -> str:
    """Get usage examples for all available tools."""
    return json.dumps({
        "intelligent_api_examples": [
            {
                "prompt": "Convert to PDF at high quality",
                "usage": "intelligent_indesign_api('Convert to PDF at high quality', source_url)"
            },
            {
                "prompt": "Export pages 1-5 as JPEG at 300 DPI",
                "usage": "intelligent_indesign_api('Export pages 1-5 as JPEG at 300 DPI', source_url)"
            },
            {
                "prompt": "Get all fonts and links",
                "usage": "intelligent_indesign_api('Get all fonts and links', source_url)"
            },
            {
                "prompt": "Merge template with CSV data",
                "usage": "intelligent_indesign_api('Merge template with CSV', template_url, {'csv': csv_url})"
            }
        ],
        "dedicated_tools": [
            {
                "name": "get_data_merge_tags",
                "description": "Extract data merge placeholders from template",
                "usage": "get_data_merge_tags(template_url)"
            },
            {
                "name": "data_merge",
                "description": "Merge CSV with template to create personalized documents",
                "usage": "data_merge(template_url, csv_url, output_format='pdf')"
            },
            {
                "name": "create_rendition_direct",
                "description": "Create PDF/JPEG/PNG rendition",
                "usage": "create_rendition_direct(source_url, 'pdf', quality='high')"
            },
            {
                "name": "get_document_info_direct",
                "description": "Get fonts, links, and pages from document",
                "usage": "get_document_info_direct(source_url)"
            }
        ],
        "workflow": [
            "1. Upload files: upload_file_and_presign('/path/to/document.indd')",
            "2a. Use intelligent API: intelligent_indesign_api('your prompt', presigned_url)",
            "2b. Or use dedicated tools: get_data_merge_tags(presigned_url)",
            "3. Get results: Returns final output automatically (if wait_for_completion=True)"
        ],
        "available_resources": {
            "technical_limits": "indesign://technical-limits",
            "asset_paths_guide": "indesign://guides/asset-paths - Input/output path management",
            "rendition_guide": "indesign://guides/rendition",
            "data_merge_guide": "indesign://guides/data-merge",
            "logging_guide": "indesign://guides/logging",
            "system_status": "indesign://status"
        },
        "documentation": "https://developer.adobe.com/firefly-services/docs/indesign-apis/guides/"
    }, indent=2)


# ============ Prompts ============

@mcp.prompt("process_indesign")
def process_indesign_prompt(source_url: str, task: str) -> str:
    """Smart prompt for any InDesign task."""
    return f"""Process this InDesign document using intelligent API:

Task: {task}
Source: {source_url}

Simply call:
intelligent_indesign_api("{task}", "{source_url}")

The RAG system will automatically:
1. Understand your task
2. Choose the right API
3. Generate the correct payload
4. Execute the request
5. Return the results

No manual payload construction needed!"""


@mcp.prompt("data_merge_workflow")
def data_merge_workflow_prompt(template_url: str, csv_url: str) -> str:
    """Complete data merge workflow with tag extraction and merge."""
    return f"""Complete data merge workflow:

Template: {template_url}
CSV Data: {csv_url}

Step 1: Extract merge tags to see what fields are available
get_data_merge_tags("{template_url}")

Step 2: Perform the merge to create personalized documents
data_merge("{template_url}", "{csv_url}", output_format="pdf")

This will:
1. Show you all data merge placeholders in the template
2. Merge the CSV data to create one output per row
3. Return download URLs for all merged documents"""


if __name__ == "__main__":
    mcp.run()

