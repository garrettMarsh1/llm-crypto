#!/usr/bin/env python3
"""
Fast Parquet Processor using ijson for large JSON files
Much faster than character-by-character parsing
"""

import ijson
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import gc
import psutil
from typing import Iterator, Dict, List
from loguru import logger
import argparse
import json
from datetime import datetime

class FastParquetProcessor:
    """Fast processor using ijson for large JSON files"""
    
    def __init__(self, chunk_size: int = 10000, max_memory_usage: float = 0.8):
        self.chunk_size = chunk_size
        self.max_memory_usage = max_memory_usage
        
    def get_memory_usage(self) -> float:
        return psutil.virtual_memory().percent / 100.0
    
    def log_memory_status(self, stage: str):
        memory_percent = self.get_memory_usage() * 100
        available_gb = psutil.virtual_memory().available / (1024**3)
        logger.info(f"{stage} - Memory: {memory_percent:.1f}% used, {available_gb:.1f}GB available")
    
    def cleanup_memory(self):
        gc.collect()
        logger.info("Memory cleanup performed")
    
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
        """Optimize DataFrame for memory efficiency"""
        
        # Convert object columns to appropriate types
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                except:
                    pass
                
                # Convert to category if low cardinality
                if df[col].nunique() / len(df) < 0.5:
                    df[col] = df[col].astype('category')
        
        # Optimize numeric columns
        for col in df.select_dtypes(include=['float64']).columns:
            try:
                df[col] = df[col].astype('float32')
            except:
                pass
        
        return df
    
    def process_json_with_ijson(self, input_file: Path, output_dir: Path) -> Dict:
        """Process large JSON file using ijson (FASTEST method)"""
        
        logger.info(f"Processing {input_file} with ijson (FAST method)...")
        self.log_memory_status("Starting ijson processing")
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Clean existing files
        for file in output_dir.glob("chunk_*.parquet"):
            file.unlink()
        
        total_samples = 0
        total_chunks = 0
        current_chunk = []
        schema = None
        
        try:
            with open(input_file, 'rb') as f:
                # Use ijson to parse JSON array items
                parser = ijson.items(f, 'item')
                
                for sample in parser:
                    # Flatten the sample
                    flattened = self.flatten_nested_dict(sample)
                    current_chunk.append(flattened)
                    
                    # Create schema from first sample
                    if schema is None:
                        schema = self.create_schema_from_sample(flattened)
                        logger.info(f"Created schema with {len(schema)} fields")
                    
                    # Process chunk when it reaches chunk_size
                    if len(current_chunk) >= self.chunk_size:
                        chunk_file = self.save_chunk_to_parquet(
                            current_chunk, total_chunks, output_dir, schema
                        )
                        
                        total_samples += len(current_chunk)
                        total_chunks += 1
                        
                        logger.info(f"Processed chunk {total_chunks}: {len(current_chunk)} samples")
                        
                        # Clear chunk and cleanup
                        current_chunk = []
                        if self.get_memory_usage() > self.max_memory_usage:
                            self.cleanup_memory()
                
                # Process remaining samples
                if current_chunk:
                    chunk_file = self.save_chunk_to_parquet(
                        current_chunk, total_chunks, output_dir, schema
                    )
                    total_samples += len(current_chunk)
                    total_chunks += 1
                    logger.info(f"Processed final chunk {total_chunks}: {len(current_chunk)} samples")
            
            # Create metadata
            metadata_file = self.create_metadata(
                total_chunks, total_samples, output_dir, input_file
            )
            
            # Calculate compression ratio
            input_size = input_file.stat().st_size
            output_size = sum(f.stat().st_size for f in output_dir.glob("chunk_*.parquet"))
            compression_ratio = (1 - output_size / input_size) * 100
            
            logger.info(f"Processing completed!")
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
            logger.error(f"Processing failed: {e}")
            raise
    
    def create_schema_from_sample(self, sample: Dict) -> pa.Schema:
        """Create optimized Parquet schema from sample data"""
        fields = []
        for key, value in sample.items():
            if isinstance(value, (int, bool)):
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
            elif isinstance(value, (float,)):
                fields.append(pa.field(key, pa.float32()))
            elif isinstance(value, str):
                fields.append(pa.field(key, pa.string()))
            else:
                # For any other type, use string as fallback
                fields.append(pa.field(key, pa.string()))
        
        return pa.schema(fields)
    
    def save_chunk_to_parquet(self, chunk: List[Dict], chunk_id: int, 
                             output_dir: Path, schema: pa.Schema) -> Path:
        """Save chunk to Parquet file"""
        
        # Create DataFrame
        df = pd.DataFrame(chunk)
        
        # Optimize DataFrame
        df = self.optimize_dataframe(df)
        
        # Convert to Arrow Table without strict schema (let Arrow infer types)
        try:
            table = pa.Table.from_pandas(df)
        except Exception as e:
            logger.warning(f"Schema mismatch, using flexible conversion: {e}")
            # Convert all columns to strings if there are type issues
            for col in df.columns:
                df[col] = df[col].astype(str)
            table = pa.Table.from_pandas(df)
        
        # Write to Parquet with compression
        output_file = output_dir / f"chunk_{chunk_id:06d}.parquet"
        pq.write_table(
            table,
            output_file,
            compression='snappy',
            use_dictionary=True,
            write_statistics=True
        )
        
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        logger.info(f"Saved chunk {chunk_id}: {output_file.name} ({file_size_mb:.1f}MB)")
        
        return output_file
    
    def create_metadata(self, total_chunks: int, total_samples: int, 
                       output_dir: Path, source_file: Path) -> Path:
        """Create metadata file"""
        
        metadata = {
            "dataset_info": {
                "total_chunks": total_chunks,
                "total_samples": total_samples,
                "chunk_size": self.chunk_size,
                "source_file": str(source_file),
                "created_at": datetime.now().isoformat(),
                "format": "parquet",
                "compression": "snappy",
                "processor": "fast_ijson"
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

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description="Fast JSON to Parquet conversion using ijson")
    parser.add_argument("--input_file", required=True, help="Input JSON file path")
    parser.add_argument("--output_dir", required=True, help="Output directory for Parquet chunks")
    parser.add_argument("--chunk_size", type=int, default=10000, help="Number of samples per chunk")
    parser.add_argument("--max_memory", type=float, default=0.8, help="Maximum memory usage threshold")
    
    args = parser.parse_args()
    
    # Setup logging
    logger.add("logs/fast_parquet_{time}.log")
    logger.info("Starting FAST Parquet conversion with ijson")
    
    # Check input file
    input_file = Path(args.input_file)
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return
    
    # Check file size
    file_size_gb = input_file.stat().st_size / (1024**3)
    logger.info(f"Input file size: {file_size_gb:.2f}GB")
    
    # Initialize processor
    processor = FastParquetProcessor(
        chunk_size=args.chunk_size,
        max_memory_usage=args.max_memory
    )
    
    # Process file
    try:
        results = processor.process_json_with_ijson(
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
