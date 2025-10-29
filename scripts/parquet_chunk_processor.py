#!/usr/bin/env python3
"""
Parquet Chunk Processor for Large LLM Training Datasets
Converts large JSON files to optimized Parquet chunks for memory-efficient training
"""

import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from pathlib import Path
import gc
import psutil
from typing import Iterator, Dict, List, Optional
from loguru import logger
import argparse
from datetime import datetime
import numpy as np

class ParquetChunkProcessor:
    """Convert large JSON datasets to optimized Parquet chunks"""
    
    def __init__(self, chunk_size: int = 10000, max_memory_usage: float = 0.8):
        self.chunk_size = chunk_size
        self.max_memory_usage = max_memory_usage
        
    def get_memory_usage(self) -> float:
        """Get current memory usage as percentage"""
        return psutil.virtual_memory().percent / 100.0
    
    def log_memory_status(self, stage: str):
        """Log current memory status"""
        memory_percent = self.get_memory_usage() * 100
        available_gb = psutil.virtual_memory().available / (1024**3)
        logger.info(f"{stage} - Memory: {memory_percent:.1f}% used, {available_gb:.1f}GB available")
        
        if memory_percent > 90:
            logger.warning("High memory usage detected! Consider reducing chunk_size")
    
    def cleanup_memory(self):
        """Force garbage collection and memory cleanup"""
        gc.collect()
        logger.info("Memory cleanup performed")
    
    def read_json_in_chunks(self, file_path: Path) -> Iterator[List[Dict]]:
        """Read large JSON file in memory-efficient chunks"""
        logger.info(f"Reading {file_path} in chunks of {self.chunk_size} samples...")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Read the opening bracket
                char = f.read(1)
                if char != '[':
                    raise ValueError("Invalid JSON array format")
                
                chunk = []
                bracket_count = 0
                in_string = False
                escape_next = False
                current_obj = ""
                
                while True:
                    char = f.read(1)
                    if not char:  # End of file
                        break
                    
                    if escape_next:
                        current_obj += char
                        escape_next = False
                        continue
                    
                    if char == '\\':
                        current_obj += char
                        escape_next = True
                        continue
                    
                    if char == '"' and not escape_next:
                        in_string = not in_string
                    
                    current_obj += char
                    
                    if not in_string:
                        if char == '{':
                            bracket_count += 1
                        elif char == '}':
                            bracket_count -= 1
                            
                            if bracket_count == 0:
                                # Complete object found
                                try:
                                    obj = json.loads(current_obj.strip().rstrip(','))
                                    chunk.append(obj)
                                    current_obj = ""
                                    
                                    if len(chunk) >= self.chunk_size:
                                        yield chunk
                                        chunk = []
                                        
                                        # Memory management
                                        if self.should_cleanup_memory():
                                            self.cleanup_memory()
                                            
                                except json.JSONDecodeError as e:
                                    logger.warning(f"Failed to parse JSON object: {e}")
                                    current_obj = ""
                                    continue
                        elif char == ']':
                            # End of array
                            if chunk:
                                yield chunk
                            break
                
                # Yield remaining chunk
                if chunk:
                    yield chunk
                    
        except Exception as e:
            logger.error(f"Error reading JSON file: {e}")
            raise
    
    def should_cleanup_memory(self) -> bool:
        """Check if memory cleanup is needed"""
        return self.get_memory_usage() > self.max_memory_usage
    
    def flatten_nested_dict(self, data: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """Flatten nested dictionaries for better Parquet storage"""
        items = []
        for k, v in data.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self.flatten_nested_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for memory efficiency and Parquet storage"""
        
        # Convert object columns to appropriate types
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert to numeric
                try:
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                except:
                    pass
                
                # Convert to category if low cardinality
                if df[col].nunique() / len(df) < 0.5:
                    df[col] = df[col].astype('category')
        
        # Optimize numeric columns
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].dtype == 'float64':
                # Try to downcast to float32
                if df[col].min() >= np.finfo(np.float32).min and df[col].max() <= np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
            elif df[col].dtype == 'int64':
                # Try to downcast to smaller int types
                if df[col].min() >= np.iinfo(np.int32).min and df[col].max() <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif df[col].min() >= np.iinfo(np.int16).min and df[col].max() <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif df[col].min() >= np.iinfo(np.int8).min and df[col].max() <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
        
        return df
    
    def create_parquet_schema(self, sample_data: Dict) -> pa.Schema:
        """Create optimized Parquet schema from sample data"""
        
        # Flatten the sample data
        flattened = self.flatten_nested_dict(sample_data)
        
        # Create schema fields
        fields = []
        for key, value in flattened.items():
            if isinstance(value, (int, np.integer)):
                if isinstance(value, bool):
                    fields.append(pa.field(key, pa.bool_()))
                elif value < 2**8:
                    fields.append(pa.field(key, pa.int8()))
                elif value < 2**16:
                    fields.append(pa.field(key, pa.int16()))
                elif value < 2**32:
                    fields.append(pa.field(key, pa.int32()))
                else:
                    fields.append(pa.field(key, pa.int64()))
            elif isinstance(value, (float, np.floating)):
                fields.append(pa.field(key, pa.float32()))
            elif isinstance(value, str):
                fields.append(pa.field(key, pa.string()))
            else:
                fields.append(pa.field(key, pa.string()))  # Fallback to string
        
        return pa.schema(fields)
    
    def convert_chunk_to_parquet(self, chunk: List[Dict], chunk_id: int, output_dir: Path, 
                                schema: Optional[pa.Schema] = None) -> Path:
        """Convert a chunk of data to Parquet format"""
        
        logger.info(f"Processing chunk {chunk_id} with {len(chunk)} samples...")
        
        # Flatten all data in chunk
        flattened_data = []
        for sample in chunk:
            flattened = self.flatten_nested_dict(sample)
            flattened_data.append(flattened)
        
        # Create DataFrame
        df = pd.DataFrame(flattened_data)
        
        # Optimize DataFrame
        df = self.optimize_dataframe(df)
        
        # Convert to Arrow Table
        table = pa.Table.from_pandas(df, schema=schema)
        
        # Write to Parquet with compression
        output_file = output_dir / f"chunk_{chunk_id:06d}.parquet"
        pq.write_table(
            table,
            output_file,
            compression='snappy',  # Fast compression
            use_dictionary=True,   # Dictionary encoding for repeated values
            write_statistics=True  # Enable statistics for better query performance
        )
        
        # Get file size
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        logger.info(f"Chunk {chunk_id} saved: {output_file.name} ({file_size_mb:.1f}MB)")
        
        return output_file
    
    def create_metadata(self, total_chunks: int, total_samples: int, 
                       output_dir: Path, source_file: Path) -> Path:
        """Create metadata file for the dataset"""
        
        metadata = {
            "dataset_info": {
                "total_chunks": total_chunks,
                "total_samples": total_samples,
                "chunk_size": self.chunk_size,
                "source_file": str(source_file),
                "created_at": datetime.now().isoformat(),
                "format": "parquet",
                "compression": "snappy"
            },
            "chunks": []
        }
        
        # Add chunk information
        for i in range(total_chunks):
            chunk_file = output_dir / f"chunk_{i:06d}.parquet"
            if chunk_file.exists():
                chunk_size = chunk_file.stat().st_size
                metadata["chunks"].append({
                    "chunk_id": i,
                    "filename": chunk_file.name,
                    "size_bytes": chunk_size,
                    "size_mb": chunk_size / (1024 * 1024)
                })
        
        # Save metadata
        metadata_file = output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved: {metadata_file}")
        return metadata_file
    
    def convert_json_to_parquet_chunks(self, input_file: Path, output_dir: Path) -> Dict:
        """Convert large JSON file to Parquet chunks"""
        
        logger.info(f"Converting {input_file} to Parquet chunks...")
        self.log_memory_status("Starting conversion")
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Clean existing files
        for file in output_dir.glob("chunk_*.parquet"):
            file.unlink()
        
        total_samples = 0
        total_chunks = 0
        schema = None
        
        try:
            # Process chunks
            for chunk in self.read_json_in_chunks(input_file):
                if not chunk:
                    continue
                
                # Create schema from first chunk
                if schema is None and chunk:
                    sample_data = chunk[0]
                    schema = self.create_parquet_schema(sample_data)
                    logger.info(f"Created schema with {len(schema)} fields")
                
                # Convert chunk to Parquet
                chunk_file = self.convert_chunk_to_parquet(
                    chunk, total_chunks, output_dir, schema
                )
                
                total_samples += len(chunk)
                total_chunks += 1
                
                # Log progress
                if total_chunks % 10 == 0:
                    self.log_memory_status(f"Processed {total_chunks} chunks")
                
                # Memory management
                if self.should_cleanup_memory():
                    self.cleanup_memory()
            
            # Create metadata
            metadata_file = self.create_metadata(
                total_chunks, total_samples, output_dir, input_file
            )
            
            # Calculate compression ratio
            input_size = input_file.stat().st_size
            output_size = sum(f.stat().st_size for f in output_dir.glob("chunk_*.parquet"))
            compression_ratio = (1 - output_size / input_size) * 100
            
            logger.info(f"Conversion completed!")
            logger.info(f"Total chunks: {total_chunks}")
            logger.info(f"Total samples: {total_samples}")
            logger.info(f"Input size: {input_size / (1024**3):.2f}GB")
            logger.info(f"Output size: {output_size / (1024**3):.2f}GB")
            logger.info(f"Compression ratio: {compression_ratio:.1f}%")
            
            return {
                "total_chunks": total_chunks,
                "total_samples": total_samples,
                "input_size_gb": input_size / (1024**3),
                "output_size_gb": output_size / (1024**3),
                "compression_ratio": compression_ratio,
                "metadata_file": metadata_file
            }
            
        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            raise

def main():
    """Main function for Parquet chunk conversion"""
    
    parser = argparse.ArgumentParser(description="Convert large JSON to Parquet chunks")
    parser.add_argument("--input_file", required=True, help="Input JSON file path")
    parser.add_argument("--output_dir", required=True, help="Output directory for Parquet chunks")
    parser.add_argument("--chunk_size", type=int, default=10000, 
                       help="Number of samples per chunk")
    parser.add_argument("--max_memory", type=float, default=0.8, 
                       help="Maximum memory usage threshold")
    
    args = parser.parse_args()
    
    # Setup logging
    logger.add("logs/parquet_conversion_{time}.log")
    logger.info("Starting Parquet chunk conversion")
    
    # Check input file
    input_file = Path(args.input_file)
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return
    
    # Check file size
    file_size_gb = input_file.stat().st_size / (1024**3)
    logger.info(f"Input file size: {file_size_gb:.2f}GB")
    
    if file_size_gb > 10:
        logger.warning("Large file detected. This may take a while...")
    
    # Initialize processor
    processor = ParquetChunkProcessor(
        chunk_size=args.chunk_size,
        max_memory_usage=args.max_memory
    )
    
    # Convert file
    try:
        results = processor.convert_json_to_parquet_chunks(
            input_file=input_file,
            output_dir=args.output_dir
        )
        
        logger.info("Conversion completed successfully!")
        logger.info(f"Results: {results}")
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        raise

if __name__ == "__main__":
    main()
