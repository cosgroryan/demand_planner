"""
Configuration management for the demand planning system.

This module handles environment variables, default settings, and configuration
validation for the demand planning system.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import logging
from rich.console import Console

console = Console()


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    host: str = "localhost"
    port: int = 5432
    database: str = "demand_planning"
    username: str = "postgres"
    password: str = ""
    connection_pool_size: int = 10


@dataclass
class APIConfig:
    """API configuration settings."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    log_level: str = "info"
    cors_origins: List[str] = field(default_factory=lambda: [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:8080"
    ])


@dataclass
class DataConfig:
    """Data processing configuration settings."""
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    series_data_dir: str = "data/processed/series"
    features_data_dir: str = "data/processed/features"
    reports_dir: str = "reports"
    backtests_dir: str = "reports/backtests"
    planning_dir: str = "reports/planning"
    max_file_size_mb: int = 1000
    compression: str = "snappy"


@dataclass
class ModelConfig:
    """Model configuration settings."""
    default_model: str = "auto"
    default_horizon: int = 13
    default_history_window: int = 52
    default_step_size: int = 4
    auto_model_selection: bool = True
    model_cache_size: int = 100
    parallel_processing: bool = True
    max_workers: int = 4


@dataclass
class InventoryConfig:
    """Inventory planning configuration settings."""
    default_service_level: float = 0.95
    default_lead_time_periods: int = 4
    default_safety_stock_multiplier: float = 1.0
    default_review_period: int = 1
    default_minimum_order_quantity: float = 0.0
    default_order_multiple: float = 1.0


@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size_mb: int = 100
    backup_count: int = 5
    console_output: bool = True
    rich_logging: bool = True


@dataclass
class UIConfig:
    """UI configuration settings."""
    streamlit_port: int = 8501
    streamlit_host: str = "localhost"
    api_base_url: str = "http://localhost:8000"
    theme: str = "light"
    page_title: str = "Demand Planner"
    page_icon: str = "📊"


class Config:
    """Main configuration class that combines all configuration sections."""
    
    def __init__(self, env_file: Optional[str] = None):
        """Initialize configuration from environment variables and defaults."""
        
        # Load environment variables from file if specified
        if env_file and Path(env_file).exists():
            self._load_env_file(env_file)
        
        # Initialize configuration sections
        self.database = DatabaseConfig()
        self.api = APIConfig()
        self.data = DataConfig()
        self.model = ModelConfig()
        self.inventory = InventoryConfig()
        self.logging = LoggingConfig()
        self.ui = UIConfig()
        
        # Load from environment variables
        self._load_from_env()
        
        # Validate configuration
        self._validate()
    
    def _load_env_file(self, env_file: str):
        """Load environment variables from a file."""
        try:
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load env file {env_file}: {e}[/yellow]")
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        
        # Database configuration
        self.database.host = os.getenv("DB_HOST", self.database.host)
        self.database.port = int(os.getenv("DB_PORT", self.database.port))
        self.database.database = os.getenv("DB_NAME", self.database.database)
        self.database.username = os.getenv("DB_USER", self.database.username)
        self.database.password = os.getenv("DB_PASSWORD", self.database.password)
        self.database.connection_pool_size = int(os.getenv("DB_POOL_SIZE", self.database.connection_pool_size))
        
        # API configuration
        self.api.host = os.getenv("API_HOST", self.api.host)
        self.api.port = int(os.getenv("API_PORT", self.api.port))
        self.api.workers = int(os.getenv("API_WORKERS", self.api.workers))
        self.api.reload = os.getenv("API_RELOAD", "false").lower() == "true"
        self.api.log_level = os.getenv("API_LOG_LEVEL", self.api.log_level)
        
        # Data configuration
        self.data.raw_data_dir = os.getenv("DATA_RAW_DIR", self.data.raw_data_dir)
        self.data.processed_data_dir = os.getenv("DATA_PROCESSED_DIR", self.data.processed_data_dir)
        self.data.series_data_dir = os.getenv("DATA_SERIES_DIR", self.data.series_data_dir)
        self.data.features_data_dir = os.getenv("DATA_FEATURES_DIR", self.data.features_data_dir)
        self.data.reports_dir = os.getenv("REPORTS_DIR", self.data.reports_dir)
        self.data.backtests_dir = os.getenv("BACKTESTS_DIR", self.data.backtests_dir)
        self.data.planning_dir = os.getenv("PLANNING_DIR", self.data.planning_dir)
        self.data.max_file_size_mb = int(os.getenv("MAX_FILE_SIZE_MB", self.data.max_file_size_mb))
        self.data.compression = os.getenv("COMPRESSION", self.data.compression)
        
        # Model configuration
        self.model.default_model = os.getenv("DEFAULT_MODEL", self.model.default_model)
        self.model.default_horizon = int(os.getenv("DEFAULT_HORIZON", self.model.default_horizon))
        self.model.default_history_window = int(os.getenv("DEFAULT_HISTORY_WINDOW", self.model.default_history_window))
        self.model.default_step_size = int(os.getenv("DEFAULT_STEP_SIZE", self.model.default_step_size))
        self.model.auto_model_selection = os.getenv("AUTO_MODEL_SELECTION", "true").lower() == "true"
        self.model.model_cache_size = int(os.getenv("MODEL_CACHE_SIZE", self.model.model_cache_size))
        self.model.parallel_processing = os.getenv("PARALLEL_PROCESSING", "true").lower() == "true"
        self.model.max_workers = int(os.getenv("MAX_WORKERS", self.model.max_workers))
        
        # Inventory configuration
        self.inventory.default_service_level = float(os.getenv("DEFAULT_SERVICE_LEVEL", self.inventory.default_service_level))
        self.inventory.default_lead_time_periods = int(os.getenv("DEFAULT_LEAD_TIME_PERIODS", self.inventory.default_lead_time_periods))
        self.inventory.default_safety_stock_multiplier = float(os.getenv("DEFAULT_SAFETY_STOCK_MULTIPLIER", self.inventory.default_safety_stock_multiplier))
        self.inventory.default_review_period = int(os.getenv("DEFAULT_REVIEW_PERIOD", self.inventory.default_review_period))
        self.inventory.default_minimum_order_quantity = float(os.getenv("DEFAULT_MIN_ORDER_QTY", self.inventory.default_minimum_order_quantity))
        self.inventory.default_order_multiple = float(os.getenv("DEFAULT_ORDER_MULTIPLE", self.inventory.default_order_multiple))
        
        # Logging configuration
        self.logging.level = os.getenv("LOG_LEVEL", self.logging.level)
        self.logging.format = os.getenv("LOG_FORMAT", self.logging.format)
        self.logging.file_path = os.getenv("LOG_FILE_PATH", self.logging.file_path)
        self.logging.max_file_size_mb = int(os.getenv("LOG_MAX_FILE_SIZE_MB", self.logging.max_file_size_mb))
        self.logging.backup_count = int(os.getenv("LOG_BACKUP_COUNT", self.logging.backup_count))
        self.logging.console_output = os.getenv("LOG_CONSOLE_OUTPUT", "true").lower() == "true"
        self.logging.rich_logging = os.getenv("LOG_RICH_LOGGING", "true").lower() == "true"
        
        # UI configuration
        self.ui.streamlit_port = int(os.getenv("STREAMLIT_PORT", self.ui.streamlit_port))
        self.ui.streamlit_host = os.getenv("STREAMLIT_HOST", self.ui.streamlit_host)
        self.ui.api_base_url = os.getenv("API_BASE_URL", self.ui.api_base_url)
        self.ui.theme = os.getenv("UI_THEME", self.ui.theme)
        self.ui.page_title = os.getenv("UI_PAGE_TITLE", self.ui.page_title)
        self.ui.page_icon = os.getenv("UI_PAGE_ICON", self.ui.page_icon)
    
    def _validate(self):
        """Validate configuration settings."""
        
        # Validate port numbers
        if not (1 <= self.api.port <= 65535):
            raise ValueError(f"Invalid API port: {self.api.port}")
        
        if not (1 <= self.ui.streamlit_port <= 65535):
            raise ValueError(f"Invalid Streamlit port: {self.ui.streamlit_port}")
        
        # Validate service level
        if not (0 <= self.inventory.default_service_level <= 1):
            raise ValueError(f"Invalid service level: {self.inventory.default_service_level}")
        
        # Validate positive numbers
        if self.model.default_horizon <= 0:
            raise ValueError(f"Invalid horizon: {self.model.default_horizon}")
        
        if self.model.default_history_window <= 0:
            raise ValueError(f"Invalid history window: {self.model.default_history_window}")
        
        if self.inventory.default_lead_time_periods <= 0:
            raise ValueError(f"Invalid lead time periods: {self.inventory.default_lead_time_periods}")
        
        # Validate log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.logging.level.upper() not in valid_log_levels:
            raise ValueError(f"Invalid log level: {self.logging.level}")
    
    def create_directories(self):
        """Create necessary directories if they don't exist."""
        
        directories = [
            self.data.raw_data_dir,
            self.data.processed_data_dir,
            self.data.series_data_dir,
            self.data.features_data_dir,
            self.data.reports_dir,
            self.data.backtests_dir,
            self.data.planning_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        
        return {
            "database": {
                "host": self.database.host,
                "port": self.database.port,
                "database": self.database.database,
                "username": self.database.username,
                "connection_pool_size": self.database.connection_pool_size
            },
            "api": {
                "host": self.api.host,
                "port": self.api.port,
                "workers": self.api.workers,
                "reload": self.api.reload,
                "log_level": self.api.log_level,
                "cors_origins": self.api.cors_origins
            },
            "data": {
                "raw_data_dir": self.data.raw_data_dir,
                "processed_data_dir": self.data.processed_data_dir,
                "series_data_dir": self.data.series_data_dir,
                "features_data_dir": self.data.features_data_dir,
                "reports_dir": self.data.reports_dir,
                "backtests_dir": self.data.backtests_dir,
                "planning_dir": self.data.planning_dir,
                "max_file_size_mb": self.data.max_file_size_mb,
                "compression": self.data.compression
            },
            "model": {
                "default_model": self.model.default_model,
                "default_horizon": self.model.default_horizon,
                "default_history_window": self.model.default_history_window,
                "default_step_size": self.model.default_step_size,
                "auto_model_selection": self.model.auto_model_selection,
                "model_cache_size": self.model.model_cache_size,
                "parallel_processing": self.model.parallel_processing,
                "max_workers": self.model.max_workers
            },
            "inventory": {
                "default_service_level": self.inventory.default_service_level,
                "default_lead_time_periods": self.inventory.default_lead_time_periods,
                "default_safety_stock_multiplier": self.inventory.default_safety_stock_multiplier,
                "default_review_period": self.inventory.default_review_period,
                "default_minimum_order_quantity": self.inventory.default_minimum_order_quantity,
                "default_order_multiple": self.inventory.default_order_multiple
            },
            "logging": {
                "level": self.logging.level,
                "format": self.logging.format,
                "file_path": self.logging.file_path,
                "max_file_size_mb": self.logging.max_file_size_mb,
                "backup_count": self.logging.backup_count,
                "console_output": self.logging.console_output,
                "rich_logging": self.logging.rich_logging
            },
            "ui": {
                "streamlit_port": self.ui.streamlit_port,
                "streamlit_host": self.ui.streamlit_host,
                "api_base_url": self.ui.api_base_url,
                "theme": self.ui.theme,
                "page_title": self.ui.page_title,
                "page_icon": self.ui.page_icon
            }
        }


def setup_logging(config: LoggingConfig):
    """Setup logging configuration."""
    
    # Configure logging level
    log_level = getattr(logging, config.level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(config.format)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    if config.console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if config.file_path:
        from logging.handlers import RotatingFileHandler
        
        file_handler = RotatingFileHandler(
            config.file_path,
            maxBytes=config.max_file_size_mb * 1024 * 1024,
            backupCount=config.backup_count
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Rich logging for better console output
    if config.rich_logging:
        try:
            from rich.logging import RichHandler
            rich_handler = RichHandler(console=console, rich_tracebacks=True)
            rich_handler.setLevel(log_level)
            root_logger.addHandler(rich_handler)
        except ImportError:
            console.print("[yellow]Rich logging not available. Install rich package for better console output.[/yellow]")


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config


def reload_config(env_file: Optional[str] = None):
    """Reload configuration from environment variables."""
    global config
    config = Config(env_file)
    return config
