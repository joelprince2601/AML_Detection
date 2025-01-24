"""Excel file processing module."""
import pandas as pd
import numpy as np
import gc
from typing import Generator, Optional
from ..utils.logging_utils import logger
from ..config.aml_config import FEATURE_COLUMNS

class ExcelProcessor:
    def __init__(self, file_path: str):
        """Initialize Excel processor."""
        self.file_path = file_path
        self.dtype_dict = None
        self.total_rows = None
    
    def _count_rows(self) -> int:
        """Count total rows in Excel file."""
        try:
            # Read only the number of rows without loading data
            xl = pd.ExcelFile(self.file_path)
            sheet = xl.book.active
            return sheet.max_row - 1  # Subtract header row
        except Exception as e:
            logger.error(f"Error counting rows: {str(e)}")
            raise

    def _optimize_dtypes(self, df: pd.DataFrame) -> dict:
        """Determine optimal data types for each column."""
        dtypes = {}
        
        for column in df.columns:
            # Skip date columns
            if 'date' in column.lower():
                continue
                
            # Get column data
            col_data = df[column]
            
            # Determine optimal numeric dtype
            if pd.api.types.is_numeric_dtype(col_data):
                if col_data.dtype == 'int64':
                    if col_data.min() >= 0:
                        if col_data.max() < 255:
                            dtypes[column] = 'uint8'
                        elif col_data.max() < 65535:
                            dtypes[column] = 'uint16'
                        else:
                            dtypes[column] = 'uint32'
                    else:
                        if col_data.min() > -128 and col_data.max() < 127:
                            dtypes[column] = 'int8'
                        elif col_data.min() > -32768 and col_data.max() < 32767:
                            dtypes[column] = 'int16'
                        else:
                            dtypes[column] = 'int32'
                elif col_data.dtype == 'float64':
                    dtypes[column] = 'float32'
            
            # Optimize string columns
            elif pd.api.types.is_string_dtype(col_data):
                if col_data.nunique() / len(col_data) < 0.5:  # If less than 50% unique values
                    dtypes[column] = 'category'
        
        return dtypes

    def read_in_chunks(self, chunk_size: int = 10000) -> Generator[pd.DataFrame, None, None]:
        """
        Read Excel file in chunks to manage memory efficiently.
        
        Args:
            chunk_size (int): Number of rows to process at once
            
        Yields:
            pd.DataFrame: Processed chunk of data
        """
        try:
            # First count total rows
            if self.total_rows is None:
                self.total_rows = self._count_rows()
                logger.info(f"Total rows in file: {self.total_rows}")
            
            # Read the Excel file
            xl = pd.ExcelFile(self.file_path)
            sheet_name = xl.sheet_names[0]  # Assume first sheet
            
            # Determine optimal data types from first chunk
            if self.dtype_dict is None:
                first_chunk = pd.read_excel(
                    xl,
                    sheet_name=sheet_name,
                    nrows=min(chunk_size, self.total_rows)
                )
                self.dtype_dict = self._optimize_dtypes(first_chunk)
                logger.info("Determined optimal data types")
            
            # Process file in chunks
            for start_row in range(0, self.total_rows, chunk_size):
                chunk = pd.read_excel(
                    xl,
                    sheet_name=sheet_name,
                    skiprows=range(1, start_row + 1),  # Skip header and previous rows
                    nrows=chunk_size,
                    dtype=self.dtype_dict
                )
                
                if chunk.empty:
                    break
                
                # Convert date columns
                date_columns = [col for col in chunk.columns if 'date' in col.lower()]
                for date_col in date_columns:
                    chunk[date_col] = pd.to_datetime(chunk[date_col])
                
                yield chunk
                
                # Clean up memory
                del chunk
                gc.collect()
            
        except Exception as e:
            logger.error(f"Error reading Excel file: {str(e)}")
            raise 