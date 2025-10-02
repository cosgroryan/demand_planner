"""
XML parser for Excel XML format files.
"""

import xml.etree.ElementTree as ET
import polars as pl
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def parse_excel_xml(xml_path: str, sample_size: Optional[int] = None) -> pl.DataFrame:
    """
    Parse Excel XML file and convert to Polars DataFrame.
    
    Args:
        xml_path: Path to the Excel XML file
        sample_size: If provided, only process this many rows (for testing)
    
    Returns:
        Polars DataFrame with the parsed data
    """
    logger.info(f"Parsing Excel XML file: {xml_path}")
    
    # Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Find the worksheet data
    worksheet = root.find('.//{urn:schemas-microsoft-com:office:spreadsheet}Worksheet')
    if worksheet is None:
        raise ValueError("No worksheet found in XML file")
    
    table = worksheet.find('.//{urn:schemas-microsoft-com:office:spreadsheet}Table')
    if table is None:
        raise ValueError("No table found in worksheet")
    
    # Extract headers from first row
    rows = table.findall('.//{urn:schemas-microsoft-com:office:spreadsheet}Row')
    if not rows:
        raise ValueError("No rows found in table")
    
    # Get headers from first row
    header_row = rows[0]
    headers = []
    for cell in header_row.findall('.//{urn:schemas-microsoft-com:office:spreadsheet}Cell'):
        data_elem = cell.find('.//{urn:schemas-microsoft-com:office:spreadsheet}Data')
        if data_elem is not None:
            headers.append(data_elem.text or "")
        else:
            headers.append("")
    
    logger.info(f"Found {len(headers)} columns: {headers}")
    
    # Extract data rows
    data_rows = []
    for i, row in enumerate(rows[1:], 1):  # Skip header row
        if sample_size and i > sample_size:
            break
            
        row_data = []
        cells = row.findall('.//{urn:schemas-microsoft-com:office:spreadsheet}Cell')
        
        # Handle sparse rows (some cells might be missing)
        cell_index = 0
        for cell in cells:
            # Get cell index if specified
            cell_ss_index = cell.get('{urn:schemas-microsoft-com:office:spreadsheet}Index')
            if cell_ss_index:
                cell_index = int(cell_ss_index) - 1  # Convert to 0-based
            
            # Fill missing cells with empty strings
            while len(row_data) < cell_index:
                row_data.append("")
            
            # Extract cell data
            data_elem = cell.find('.//{urn:schemas-microsoft-com:office:spreadsheet}Data')
            if data_elem is not None:
                cell_value = data_elem.text or ""
                # Handle datetime values
                if data_elem.get('{urn:schemas-microsoft-com:office:spreadsheet}Type') == 'DateTime':
                    try:
                        # Parse datetime and convert to string
                        dt = datetime.fromisoformat(cell_value.replace('Z', '+00:00'))
                        cell_value = dt.strftime('%Y-%m-%d')
                    except:
                        pass  # Keep original value if parsing fails
                row_data.append(cell_value)
            else:
                row_data.append("")
            
            cell_index += 1
        
        # Ensure row has same length as headers
        while len(row_data) < len(headers):
            row_data.append("")
        
        # Truncate if too long
        row_data = row_data[:len(headers)]
        
        data_rows.append(row_data)
        
        if i % 10000 == 0:
            logger.info(f"Processed {i} rows...")
    
    logger.info(f"Parsed {len(data_rows)} data rows")
    
    # Create DataFrame
    df = pl.DataFrame(data_rows, schema=headers)
    
    # Clean up column names
    df = df.rename({col: col.strip() for col in df.columns})
    
    return df


def xml_to_parquet(xml_path: str, output_path: str, sample_size: Optional[int] = None) -> None:
    """
    Convert Excel XML file to Parquet format.
    
    Args:
        xml_path: Path to the Excel XML file
        output_path: Path for output Parquet file
        sample_size: If provided, only process this many rows (for testing)
    """
    df = parse_excel_xml(xml_path, sample_size)
    
    # Ensure output directory exists
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to parquet
    df.write_parquet(output_path, compression='snappy')
    logger.info(f"Saved {len(df)} rows to {output_path}")
