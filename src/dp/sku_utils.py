"""
SKU utilities for handling Parent-Style-Colour-Size structure.

This module provides functions to parse, validate, and reconstruct SKUs
in the format: Parent-Style-Colour-Size (e.g., 100007-1004-001-S)
"""

import re
from typing import List, Dict, Any, Optional, Tuple
import polars as pl


def parse_sku(sku_string: str) -> Dict[str, str]:
    """
    Parse a SKU string into its components.
    
    Args:
        sku_string: SKU in format Parent-Style-Colour-Size
    
    Returns:
        Dictionary with parsed components
    """
    if not sku_string or sku_string == "null":
        return {"parent": "", "style": "", "colour": "", "size": "", "full_sku": ""}
    
    # Clean the SKU string
    sku_clean = str(sku_string).strip()
    
    # Split by hyphen
    parts = sku_clean.split('-')
    
    if len(parts) == 4:
        # Full format: Parent-Style-Colour-Size
        return {
            "parent": parts[0],
            "style": parts[1], 
            "colour": parts[2],
            "size": parts[3],
            "full_sku": sku_clean
        }
    elif len(parts) == 3:
        # Missing size: Parent-Style-Colour
        return {
            "parent": parts[0],
            "style": parts[1],
            "colour": parts[2], 
            "size": "",
            "full_sku": sku_clean
        }
    elif len(parts) == 2:
        # Only Parent-Style
        return {
            "parent": parts[0],
            "style": parts[1],
            "colour": "",
            "size": "",
            "full_sku": sku_clean
        }
    else:
        # Single part or invalid format
        return {
            "parent": sku_clean,
            "style": "",
            "colour": "",
            "size": "",
            "full_sku": sku_clean
        }


def reconstruct_sku_from_fields(
    style: str = "",
    colour: str = "",
    gender: str = "",
    internal_id: str = "",
    style_colour_code: str = ""
) -> str:
    """
    Reconstruct a proper SKU in 6-4-3-size format from available fields.
    
    Args:
        style: Style field
        colour: Colour field (may contain full SKU)
        gender: Gender field
        internal_id: Internal ID field
        style_colour_code: Style colour code field
    
    Returns:
        Reconstructed SKU string in format: 100007-1004-001-S
    """
    # Try to extract SKU from colour field first (it often contains the full SKU)
    if colour and colour != "null" and "-" in str(colour):
        colour_clean = str(colour).strip()
        # Check if it looks like a proper SKU (has 3+ parts with hyphens)
        parts = colour_clean.split('-')
        if len(parts) >= 3:
            # Validate and format the SKU
            formatted_sku = format_sku_to_standard(colour_clean)
            if formatted_sku:
                return formatted_sku
    
    # Try to extract from internal ID
    if internal_id and internal_id != "null":
        internal_clean = str(internal_id).strip()
        # Look for patterns like "100557-1228-021-XXL" in the internal ID
        sku_pattern = r'(\d{6}-\d{4}-\d{3}-[A-Z]{2,3})'
        match = re.search(sku_pattern, internal_clean)
        if match:
            return match.group(1)
    
    # Try to construct from style and colour code
    if style and style_colour_code and style != "null" and style_colour_code != "null":
        style_clean = str(style).strip()
        code_clean = str(style_colour_code).strip()
        
        # If style_colour_code looks like a partial SKU
        if "-" in code_clean:
            formatted_sku = format_sku_to_standard(code_clean)
            if formatted_sku:
                return formatted_sku
        
        # Try to construct: use style_colour_code as base
        constructed = f"{code_clean}-{style_clean}"
        formatted_sku = format_sku_to_standard(constructed)
        if formatted_sku:
            return formatted_sku
    
    # Fallback: use internal ID or create a simple identifier
    if internal_id and internal_id != "null":
        # Extract meaningful parts from internal ID
        internal_clean = str(internal_id).strip()
        # Look for any numeric patterns that could be SKU components
        numeric_pattern = r'\d{6}-\d{4}-\d{3}'
        match = re.search(numeric_pattern, internal_clean)
        if match:
            # Add a default size
            return f"{match.group(0)}-S"
    
    # Last resort: create a properly formatted SKU
    # Generate a 6-digit parent, 4-digit style, 3-digit colour, and size
    parent = f"{hash(str(style) + str(colour)) % 1000000:06d}"
    style_num = f"{hash(str(style)) % 10000:04d}"
    colour_num = f"{hash(str(colour)) % 1000:03d}"
    size = "S"  # Default size
    
    return f"{parent}-{style_num}-{colour_num}-{size}"


def normalize_size(size: str) -> str:
    """
    Normalize size to standard format.
    
    Args:
        size: Size string
    
    Returns:
        Normalized size string
    """
    if not size:
        return "S"
    
    size_upper = str(size).upper().strip()
    
    # Common size mappings
    size_mappings = {
        "XXL": "XL",
        "XXXL": "XL", 
        "XS": "S",
        "XXS": "S",
        "XXXS": "S"
    }
    
    return size_mappings.get(size_upper, size_upper)


def format_sku_to_standard(sku_string: str, include_size: bool = True) -> Optional[str]:
    """
    Format a SKU string to the standard 6-4-3 format (with or without size).
    
    Args:
        sku_string: Input SKU string
        include_size: Whether to include size component (default: True for full SKU)
    
    Returns:
        Formatted SKU string or None if invalid
    """
    if not sku_string:
        return None
    
    # Clean the input
    sku_clean = str(sku_string).strip()
    
    # Split by hyphen
    parts = sku_clean.split('-')
    
    if len(parts) == 4:
        # Already in correct format
        parent, style, colour, size = parts
        if (len(parent) == 6 and parent.isdigit() and
            len(style) == 4 and style.isdigit() and
            len(colour) == 3 and colour.isdigit() and
            len(size) <= 3):
            if include_size:
                # Normalize size (XXL -> XL, etc.)
                size_normalized = normalize_size(size)
                return f"{parent}-{style}-{colour}-{size_normalized}"
            else:
                # Return without size for forecasting
                return f"{parent}-{style}-{colour}"
    elif len(parts) == 3:
        # Missing size
        parent, style, colour = parts
        if (len(parent) == 6 and parent.isdigit() and
            len(style) == 4 and style.isdigit() and
            len(colour) == 3 and colour.isdigit()):
            if include_size:
                return f"{parent}-{style}-{colour}-S"
            else:
                return f"{parent}-{style}-{colour}"
    
    # Try to extract numeric parts and reconstruct
    numeric_parts = []
    for part in parts:
        if part.isdigit():
            numeric_parts.append(part)
    
    if len(numeric_parts) >= 3:
        parent = numeric_parts[0].zfill(6)[:6]  # Ensure 6 digits
        style = numeric_parts[1].zfill(4)[:4]   # Ensure 4 digits
        colour = numeric_parts[2].zfill(3)[:3]  # Ensure 3 digits
        
        if include_size:
            size = "S"  # Default size
            return f"{parent}-{style}-{colour}-{size}"
        else:
            return f"{parent}-{style}-{colour}"
    
    return None


def extract_sku_components(df: pl.DataFrame) -> pl.DataFrame:
    """
    Extract and parse SKU components from a DataFrame.
    
    Args:
        df: DataFrame with SKU-related columns
    
    Returns:
        DataFrame with parsed SKU components
    """
    # Reconstruct proper SKUs
    df_with_sku = df.with_columns([
        pl.struct([
            pl.col("Style"),
            pl.col("Colour"), 
            pl.col("Gender"),
            pl.col("Internal ID"),
            pl.col("StyleColourCode")
        ]).map_elements(
            lambda x: reconstruct_sku_from_fields(
                x["Style"], x["Colour"], x["Gender"], 
                x["Internal ID"], x["StyleColourCode"]
            ),
            return_dtype=pl.Utf8
        ).alias("SKU_Reconstructed")
    ])
    
    # Parse SKU components
    df_with_components = df_with_sku.with_columns([
        pl.col("SKU_Reconstructed").map_elements(
            lambda x: parse_sku(x)["parent"],
            return_dtype=pl.Utf8
        ).alias("SKU_Parent"),
        
        pl.col("SKU_Reconstructed").map_elements(
            lambda x: parse_sku(x)["style"],
            return_dtype=pl.Utf8
        ).alias("SKU_Style"),
        
        pl.col("SKU_Reconstructed").map_elements(
            lambda x: parse_sku(x)["colour"],
            return_dtype=pl.Utf8
        ).alias("SKU_Colour"),
        
        pl.col("SKU_Reconstructed").map_elements(
            lambda x: parse_sku(x)["size"],
            return_dtype=pl.Utf8
        ).alias("SKU_Size")
    ])
    
    # Replace the original SKU column with the reconstructed one
    df_final = df_with_components.with_columns([
        pl.col("SKU_Reconstructed").alias("SKU")
    ]).drop("SKU_Reconstructed")
    
    return df_final


def create_forecast_sku(df: pl.DataFrame) -> pl.DataFrame:
    """
    Create SKU aggregation key without sizes for forecasting.
    
    Args:
        df: DataFrame with SKU components
    
    Returns:
        DataFrame with forecast SKU (parent-style-colour without size)
    """
    # Create forecast SKU by combining parent, style, and colour (no size)
    df_with_forecast_sku = df.with_columns([
        (pl.col("SKU_Parent") + pl.lit("-") + 
         pl.col("SKU_Style") + pl.lit("-") + 
         pl.col("SKU_Colour")).alias("SKU_Forecast")
    ])
    
    return df_with_forecast_sku


def get_sku_hierarchy(df: pl.DataFrame) -> Dict[str, List[str]]:
    """
    Get the SKU hierarchy for filtering and aggregation.
    
    Args:
        df: DataFrame with SKU components
    
    Returns:
        Dictionary with hierarchy levels
    """
    hierarchy = {
        "parents": df.select("SKU_Parent").unique().to_series().to_list(),
        "styles": df.select("SKU_Style").unique().to_series().to_list(),
        "colours": df.select("SKU_Colour").unique().to_series().to_list(),
        "sizes": df.select("SKU_Size").unique().to_series().to_list(),
        "full_skus": df.select("SKU").unique().to_series().to_list()
    }
    
    # Remove empty values
    for key in hierarchy:
        hierarchy[key] = [x for x in hierarchy[key] if x and x != ""]
    
    return hierarchy


def filter_by_sku_level(
    df: pl.DataFrame, 
    level: str, 
    value: str
) -> pl.DataFrame:
    """
    Filter DataFrame by SKU hierarchy level.
    
    Args:
        df: DataFrame with SKU components
        level: Level to filter by ("parent", "style", "colour", "size", "full")
        value: Value to filter by
    
    Returns:
        Filtered DataFrame
    """
    if level == "parent":
        return df.filter(pl.col("SKU_Parent") == value)
    elif level == "style":
        return df.filter(pl.col("SKU_Style") == value)
    elif level == "colour":
        return df.filter(pl.col("SKU_Colour") == value)
    elif level == "size":
        return df.filter(pl.col("SKU_Size") == value)
    elif level == "full":
        return df.filter(pl.col("SKU") == value)
    else:
        raise ValueError(f"Invalid level: {level}. Must be one of: parent, style, colour, size, full")


def create_sku_aggregation_key(
    df: pl.DataFrame,
    levels: List[str] = ["parent", "style", "colour", "size"]
) -> pl.DataFrame:
    """
    Create aggregation keys based on SKU hierarchy levels.
    
    Args:
        df: DataFrame with SKU components
        levels: List of levels to include in aggregation key
    
    Returns:
        DataFrame with aggregation key column
    """
    # Create aggregation key based on specified levels
    key_parts = []
    for level in levels:
        if level == "parent":
            key_parts.append(pl.col("SKU_Parent"))
        elif level == "style":
            key_parts.append(pl.col("SKU_Style"))
        elif level == "colour":
            key_parts.append(pl.col("SKU_Colour"))
        elif level == "size":
            key_parts.append(pl.col("SKU_Size"))
    
    if key_parts:
        # Join the parts with hyphens
        aggregation_key = key_parts[0]
        for part in key_parts[1:]:
            aggregation_key = aggregation_key + pl.lit("-") + part
        
        df_with_key = df.with_columns([
            aggregation_key.alias("SKU_Aggregation_Key")
        ])
    else:
        # Use full SKU if no levels specified
        df_with_key = df.with_columns([
            pl.col("SKU").alias("SKU_Aggregation_Key")
        ])
    
    return df_with_key
