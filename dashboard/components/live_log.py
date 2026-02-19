"""Live Log Component for Streamlit applications.

Provides a real-time scrolling log display with color-coded entries.
"""

import streamlit as st
from datetime import datetime
from typing import Optional, List
from dataclasses import dataclass, field
from enum import Enum


class LogLevel(Enum):
    """Log level enumeration with associated emojis."""
    DEBUG = ("🔍", "#6c757d")
    INFO = ("ℹ️", "#0dcaf0")
    SUCCESS = ("✅", "#198754")
    WARNING = ("⚠️", "#ffc107")
    ERROR = ("❌", "#dc3545")
    PROGRESS = ("📊", "#6f42c1")
    STATION = ("📍", "#fd7e14")
    TRAINING = ("🧠", "#0d6efd")


@dataclass
class LogEntry:
    """A single log entry."""
    timestamp: datetime
    level: LogLevel
    message: str
    
    def format(self) -> str:
        """Format the log entry as a string."""
        time_str = self.timestamp.strftime("%H:%M:%S")
        emoji = self.level.value[0]
        return f"[{time_str}] {emoji} {self.message}"


class LiveLogManager:
    """
    Manages a live log buffer with Streamlit integration.
    
    Usage:
        log_manager = LiveLogManager(max_entries=100)
        log_manager.info("Starting process...")
        log_manager.success("Process complete!")
        log_manager.render(placeholder)
    """
    
    def __init__(self, max_entries: int = 100, session_key: str = "live_log"):
        self.max_entries = max_entries
        self.session_key = session_key
        
        # Initialize session state
        if self.session_key not in st.session_state:
            st.session_state[self.session_key] = []
    
    @property
    def entries(self) -> List[LogEntry]:
        """Get log entries from session state."""
        return st.session_state.get(self.session_key, [])
    
    def _add_entry(self, level: LogLevel, message: str):
        """Add a log entry."""
        entry = LogEntry(
            timestamp=datetime.now(),
            level=level,
            message=message
        )
        
        logs = self.entries.copy()
        logs.append(entry)
        
        # Keep only last max_entries
        if len(logs) > self.max_entries:
            logs = logs[-self.max_entries:]
        
        st.session_state[self.session_key] = logs
    
    def debug(self, message: str):
        """Log a debug message."""
        self._add_entry(LogLevel.DEBUG, message)
    
    def info(self, message: str):
        """Log an info message."""
        self._add_entry(LogLevel.INFO, message)
    
    def success(self, message: str):
        """Log a success message."""
        self._add_entry(LogLevel.SUCCESS, message)
    
    def warning(self, message: str):
        """Log a warning message."""
        self._add_entry(LogLevel.WARNING, message)
    
    def error(self, message: str):
        """Log an error message."""
        self._add_entry(LogLevel.ERROR, message)
    
    def progress(self, message: str):
        """Log a progress message."""
        self._add_entry(LogLevel.PROGRESS, message)
    
    def station(self, message: str):
        """Log a station-related message."""
        self._add_entry(LogLevel.STATION, message)
    
    def training(self, message: str):
        """Log a training-related message."""
        self._add_entry(LogLevel.TRAINING, message)
    
    def clear(self):
        """Clear all log entries."""
        st.session_state[self.session_key] = []
    
    def get_formatted_logs(self, last_n: Optional[int] = None) -> str:
        """Get formatted log string."""
        entries = self.entries
        if last_n:
            entries = entries[-last_n:]
        return "\n".join(entry.format() for entry in entries)
    
    def render(self, placeholder=None, height: int = 300, expanded: bool = True):
        """
        Render the log display.
        
        Args:
            placeholder: Streamlit placeholder to render into (optional)
            height: Height of the log container in pixels
            expanded: Whether the log is initially expanded
        """
        log_text = self.get_formatted_logs()
        
        if not log_text:
            log_text = "Waiting for logs..."
        
        # CSS for log styling
        log_css = """
        <style>
        .live-log-container {
            background-color: #1e1e1e;
            color: #d4d4d4;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 12px;
            padding: 10px;
            border-radius: 5px;
            overflow-y: auto;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        </style>
        """
        
        # Render
        if placeholder is not None:
            with placeholder:
                with st.expander("📋 Live Training Log", expanded=expanded):
                    st.markdown(log_css, unsafe_allow_html=True)
                    st.code(log_text, language=None)
        else:
            with st.expander("📋 Live Training Log", expanded=expanded):
                st.markdown(log_css, unsafe_allow_html=True)
                st.code(log_text, language=None)
    
    def render_inline(self, placeholder):
        """Render logs directly into a placeholder without expander."""
        log_text = self.get_formatted_logs()
        
        if not log_text:
            log_text = "Waiting for logs..."
        
        placeholder.code(log_text, language=None)


def create_progress_section():
    """
    Create a comprehensive progress section with multiple indicators.
    
    Returns:
        dict with placeholders for various progress elements
    """
    progress_container = st.container()
    
    with progress_container:
        st.markdown("### 📊 Training Progress")
        
        # Station Progress
        col1, col2 = st.columns([4, 1])
        with col1:
            station_progress = st.progress(0, text="Preparing stations...")
        with col2:
            station_counter = st.empty()
        
        # Current operation
        current_operation = st.empty()
        
        st.markdown("---")
        
        # Epoch Progress
        col1, col2 = st.columns([4, 1])
        with col1:
            epoch_progress = st.progress(0, text="Waiting for training to start...")
        with col2:
            epoch_counter = st.empty()
        
        # Metrics row
        metrics_row = st.empty()
        
        # Loss chart
        loss_chart = st.empty()
    
    return {
        'station_progress': station_progress,
        'station_counter': station_counter,
        'current_operation': current_operation,
        'epoch_progress': epoch_progress,
        'epoch_counter': epoch_counter,
        'metrics_row': metrics_row,
        'loss_chart': loss_chart
    }


class TrainingProgressTracker:
    """
    Tracks and displays training progress with live updates.
    """
    
    def __init__(self, log_manager: LiveLogManager, placeholders: dict):
        self.log = log_manager
        self.ph = placeholders
        self.start_time = None
        self.total_stations = 0
        self.current_station = 0
        self.total_epochs = 0
        self.current_epoch = 0
    
    def start_preparation(self, total_stations: int):
        """Mark start of station preparation phase."""
        import time
        self.start_time = time.time()
        self.total_stations = total_stations
        self.current_station = 0
        
        self.log.info(f"Starting preparation of {total_stations} stations...")
        self._update_station_progress()
    
    def prepare_station(self, station_idx: int, station_name: str, details: dict = None):
        """Update progress for a station being prepared."""
        self.current_station = station_idx + 1
        
        # Update progress bar
        progress = self.current_station / self.total_stations
        self.ph['station_progress'].progress(
            progress, 
            text=f"Preparing station {self.current_station}/{self.total_stations}"
        )
        
        # Update counter
        self.ph['station_counter'].metric(
            "Stations", 
            f"{self.current_station}/{self.total_stations}"
        )
        
        # Update current operation
        self.ph['current_operation'].info(f"📍 Processing: **{station_name}**")
        
        # Log details
        self.log.station(f"Preparing station {self.current_station}/{self.total_stations}: {station_name}")
        
        if details:
            if 'rows' in details:
                self.log.info(f"  → {details['rows']:,} rows loaded")
            if 'missing' in details:
                self.log.info(f"  → {details['missing']} missing values handled")
            if 'split' in details:
                self.log.info(f"  → Split: train={details['split'][0]}, val={details['split'][1]}, test={details['split'][2]}")
    
    def station_complete(self, station_name: str, duration: float):
        """Mark a station as complete."""
        self.log.success(f"Station {station_name} prepared in {duration:.1f}s")
    
    def start_training(self, total_epochs: int, model_name: str):
        """Mark start of training phase."""
        import time
        self.start_time = time.time()
        self.total_epochs = total_epochs
        self.current_epoch = 0
        
        self.log.training(f"Starting {model_name} training ({total_epochs} epochs)...")
        self.ph['station_progress'].progress(1.0, text="✅ All stations prepared")
        self.ph['current_operation'].success("🚀 **Training in progress...**")
    
    def update_epoch(self, epoch: int, train_loss: float = None, val_loss: float = None):
        """Update epoch progress."""
        self.current_epoch = epoch
        
        # Update progress bar
        progress = epoch / max(self.total_epochs, 1)
        self.ph['epoch_progress'].progress(
            progress,
            text=f"Epoch {epoch}/{self.total_epochs}"
        )
        
        # Update counter
        self.ph['epoch_counter'].metric("Epoch", f"{epoch}/{self.total_epochs}")
        
        # Log
        log_msg = f"Epoch {epoch}/{self.total_epochs}"
        if train_loss is not None:
            log_msg += f" | train_loss={train_loss:.4f}"
        if val_loss is not None:
            log_msg += f" | val_loss={val_loss:.4f}"
        
        self.log.training(log_msg)
        
        # Update metrics
        if train_loss is not None or val_loss is not None:
            with self.ph['metrics_row'].container():
                cols = st.columns(4)
                cols[0].metric("Epoch", f"{epoch}/{self.total_epochs}")
                if train_loss is not None:
                    cols[1].metric("Train Loss", f"{train_loss:.4f}")
                if val_loss is not None:
                    cols[2].metric("Val Loss", f"{val_loss:.4f}")
                
                # ETA calculation
                import time
                if self.start_time and epoch > 0:
                    elapsed = time.time() - self.start_time
                    avg_epoch_time = elapsed / epoch
                    remaining = avg_epoch_time * (self.total_epochs - epoch)
                    mins, secs = divmod(int(remaining), 60)
                    cols[3].metric("ETA", f"{mins}m {secs}s")
    
    def training_complete(self, metrics: dict = None):
        """Mark training as complete."""
        import time
        total_time = time.time() - self.start_time if self.start_time else 0
        mins, secs = divmod(int(total_time), 60)
        
        self.ph['epoch_progress'].progress(1.0, text="✅ Training complete!")
        self.ph['current_operation'].success(f"✅ **Training complete in {mins}m {secs}s**")
        
        self.log.success(f"Training complete! Total time: {mins}m {secs}s")
        
        if metrics:
            metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items() if v is not None)
            self.log.success(f"Final metrics: {metrics_str}")
