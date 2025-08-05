import logging
import os
import tempfile
import hashlib
import mimetypes
from typing import Optional, Dict, Any, BinaryIO
from pathlib import Path
import aiofiles
import asyncio
from ..core.config import settings

logger = logging.getLogger(__name__)

class FileHandler:
    """Utility class for handling file operations"""
    
    def __init__(self):
        self.temp_dir = tempfile.gettempdir()
        self.max_file_size = settings.MAX_FILE_SIZE
        self.allowed_extensions = settings.ALLOWED_EXTENSIONS
        
    async def validate_file(self, file_path: str, file_content: bytes = None) -> Dict[str, Any]:
        """Validate file size, type, and other constraints"""
        try:
            validation_result = {
                "is_valid": True,
                "errors": [],
                "warnings": [],
                "file_info": {}
            }
            
            # Check file extension
            file_extension = Path(file_path).suffix.lower()
            if file_extension not in self.allowed_extensions:
                validation_result["is_valid"] = False
                validation_result["errors"].append(f"File type {file_extension} not allowed")
            
            # Check file size if content is provided
            if file_content:
                file_size = len(file_content)
                if file_size > self.max_file_size:
                    validation_result["is_valid"] = False
                    validation_result["errors"].append(f"File size {file_size} exceeds maximum {self.max_file_size}")
                
                validation_result["file_info"]["size"] = file_size
            
            # Get MIME type
            mime_type, _ = mimetypes.guess_type(file_path)
            validation_result["file_info"]["mime_type"] = mime_type
            validation_result["file_info"]["extension"] = file_extension
            
            # Check if file appears to be corrupted (basic check)
            if file_content and len(file_content) < 100:
                validation_result["warnings"].append("File appears to be very small, possibly corrupted")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating file {file_path}: {str(e)}")
            return {
                "is_valid": False,
                "errors": [f"Validation error: {str(e)}"],
                "warnings": [],
                "file_info": {}
            }
    
    async def save_temp_file(self, file_content: bytes, 
                           original_filename: str = None) -> str:
        """Save file content to temporary location"""
        try:
            # Generate unique filename
            file_hash = hashlib.md5(file_content).hexdigest()[:16]
            
            if original_filename:
                extension = Path(original_filename).suffix
                temp_filename = f"{file_hash}_{original_filename}"
            else:
                extension = ".tmp"
                temp_filename = f"{file_hash}{extension}"
            
            temp_path = os.path.join(self.temp_dir, temp_filename)
            
            # Write file asynchronously
            async with aiofiles.open(temp_path, 'wb') as f:
                await f.write(file_content)
            
            logger.info(f"Saved temporary file: {temp_path}")
            return temp_path
            
        except Exception as e:
            logger.error(f"Error saving temporary file: {str(e)}")
            raise
    
    async def read_file_async(self, file_path: str) -> bytes:
        """Read file content asynchronously"""
        try:
            async with aiofiles.open(file_path, 'rb') as f:
                content = await f.read()
            return content
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            raise
    
    def cleanup_temp_file(self, file_path: str) -> bool:
        """Clean up temporary file"""
        try:
            if os.path.exists(file_path) and file_path.startswith(self.temp_dir):
                os.remove(file_path)
                logger.info(f"Cleaned up temporary file: {file_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error cleaning up file {file_path}: {str(e)}")
            return False
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get comprehensive file information"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            stat_info = os.stat(file_path)
            path_obj = Path(file_path)
            
            return {
                "filename": path_obj.name,
                "extension": path_obj.suffix.lower(),
                "size": stat_info.st_size,
                "size_mb": round(stat_info.st_size / (1024 * 1024), 2),
                "created_time": stat_info.st_ctime,
                "modified_time": stat_info.st_mtime,
                "mime_type": mimetypes.guess_type(file_path)[0],
                "is_readable": os.access(file_path, os.R_OK),
                "absolute_path": os.path.abspath(file_path)
            }
            
        except Exception as e:
            logger.error(f"Error getting file info for {file_path}: {str(e)}")
            return {"error": str(e)}
    
    async def create_file_hash(self, file_content: bytes, 
                             algorithm: str = "md5") -> str:
        """Create hash of file content"""
        try:
            if algorithm == "md5":
                hash_obj = hashlib.md5()
            elif algorithm == "sha256":
                hash_obj = hashlib.sha256()
            else:
                raise ValueError(f"Unsupported hash algorithm: {algorithm}")
            
            # Process in chunks for large files
            chunk_size = 8192
            for i in range(0, len(file_content), chunk_size):
                chunk = file_content[i:i + chunk_size]
                hash_obj.update(chunk)
            
            return hash_obj.hexdigest()
            
        except Exception as e:
            logger.error(f"Error creating file hash: {str(e)}")
            raise
    
    def is_safe_path(self, file_path: str) -> bool:
        """Check if file path is safe (no directory traversal)"""
        try:
            # Resolve path and check if it's within allowed directories
            resolved_path = os.path.abspath(file_path)
            
            # Check for directory traversal attempts
            if ".." in file_path or file_path.startswith("/"):
                return False
            
            # Check if path is within temp directory (for temp files)
            if resolved_path.startswith(os.path.abspath(self.temp_dir)):
                return True
            
            # Add other allowed directories as needed
            return False
            
        except Exception as e:
            logger.error(f"Error checking path safety: {str(e)}")
            return False
    
    async def batch_cleanup(self, file_paths: list) -> Dict[str, Any]:
        """Clean up multiple files"""
        try:
            results = {
                "cleaned": [],
                "failed": [],
                "total_processed": len(file_paths)
            }
            
            for file_path in file_paths:
                try:
                    if self.cleanup_temp_file(file_path):
                        results["cleaned"].append(file_path)
                    else:
                        results["failed"].append(file_path)
                except Exception as e:
                    logger.error(f"Error cleaning {file_path}: {str(e)}")
                    results["failed"].append(file_path)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch cleanup: {str(e)}")
            return {
                "cleaned": [],
                "failed": file_paths,
                "total_processed": len(file_paths),
                "error": str(e)
            }
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage usage statistics"""
        try:
            import shutil
            
            # Get temp directory stats
            temp_usage = shutil.disk_usage(self.temp_dir)
            
            # Count temp files created by this app
            temp_files = []
            for file in os.listdir(self.temp_dir):
                if file.startswith("llm_"):  # Assuming our temp files have this prefix
                    temp_files.append(file)
            
            return {
                "temp_directory": self.temp_dir,
                "total_space": temp_usage.total,
                "used_space": temp_usage.used,
                "free_space": temp_usage.free,
                "temp_files_count": len(temp_files),
                "max_file_size": self.max_file_size,
                "allowed_extensions": list(self.allowed_extensions)
            }
            
        except Exception as e:
            logger.error(f"Error getting storage stats: {str(e)}")
            return {"error": str(e)}

