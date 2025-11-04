#!/usr/bin/env python3
"""
Raspberry Pi 3B - Wireless Agriculture Neuromorphic Receiver

Author: MarvelMathesh
Co-Author: Sai Siddharth
Date: 2025

Installation:
    sudo apt-get install python3-dev libboost-python-dev python3-pip python3-rpi.gpio
    git clone https://github.com/nRF24/RF24.git
    cd RF24 && ./configure && make && sudo make install
    cd pyRF24 && python3 setup.py build && sudo python3 setup.py install
"""

import time
import csv
import struct
import threading
import queue
import argparse
from datetime import datetime
from typing import Optional, List, Dict, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# RF24 library import
try:
    from RF24 import RF24, RF24_PA_LOW, RF24_250KBPS
    HAS_RF24 = True
except ImportError as e:
    print(f"[WARNING] RF24 library not found: {e}")
    print("\nInstallation instructions available in file header")
    HAS_RF24 = False


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """System-wide configuration constants"""
    
    # Radio settings
    RF_CE_PIN = 22
    RF_CSN_PIN = 0  # SPI CE0
    RF_CHANNEL = 76  # 2.476 GHz
    RF_ADDRESS = bytes([0x41, 0x47, 0x52, 0x49, 0x43])  # 'AGRIC'
    RF_PAYLOAD_SIZE = 16
    
    # Data processing
    SENSOR_NAMES = ['temp', 'humid', 'tds', 'soil']
    ENCODING_NAMES = ['raw_data', 'temporal', 'rate', 'population']
    
    # Visualization
    SPIKE_HISTORY_DURATION = 10.0  # seconds
    RAW_VALUE_HISTORY_DURATION = 60.0  # seconds
    VISUALIZATION_UPDATE_INTERVAL = 100  # milliseconds
    PLOT_FIGURE_SIZE = (20, 10)
    
    # Threading
    RECEIVER_LOOP_DELAY = 0.001  # seconds
    MAX_CONSECUTIVE_ERRORS = 10
    
    # Colors (hex RGB)
    COLORS = {
        'temp': '#FF4444',
        'humid': '#44FF44',
        'tds': '#4444FF',
        'soil': '#FFAA44'
    }


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SpikeEvent:
    """Container for spike event data"""
    sensor_id: str
    timestamp: int  # Pico timestamp (ms)
    encoding_type: str
    neuron_id: int
    polarity: float
    received_time: float = field(default_factory=time.time)
    
    @property
    def is_raw_data(self) -> bool:
        """Check if this is raw sensor data"""
        return self.encoding_type == 'raw_data'
    
    @property
    def latency_ms(self) -> float:
        """Calculate transmission latency in milliseconds"""
        return (self.received_time - (self.timestamp / 1000.0)) * 1000.0


@dataclass
class ReceiverStatistics:
    """Receiver performance statistics"""
    packets_received: int = 0
    packets_lost: int = 0
    parse_errors: int = 0
    
    @property
    def total_packets(self) -> int:
        return self.packets_received + self.packets_lost
    
    @property
    def success_rate(self) -> float:
        if self.total_packets == 0:
            return 0.0
        return 100.0 * self.packets_received / self.total_packets


# =============================================================================
# RADIO RECEIVER
# =============================================================================

class RF24Receiver:
    """Modern NRF24L01+ wireless receiver using RF24 library"""
    
    def __init__(self):
        """Initialize receiver (radio configuration done in connect())"""
        self.radio: Optional[RF24] = None
        self.running = False
        self.spike_queue = queue.Queue()
        self.receive_thread: Optional[threading.Thread] = None
        self.stats = ReceiverStatistics()
    
    def connect(self) -> bool:
        """
        Initialize and configure RF24 radio module
        
        Returns:
            True if successful, False otherwise
        """
        if not HAS_RF24:
            print("[ERROR] RF24 library not available")
            return False
        
        try:
            # Initialize radio hardware
            self.radio = RF24(Config.RF_CE_PIN, Config.RF_CSN_PIN)
            
            if not self.radio.begin():
                print("[ERROR] Radio hardware not responding")
                return False
            
            # Configure radio parameters
            self.radio.setAddressWidth(5)
            self.radio.setPALevel(RF24_PA_LOW)
            self.radio.setDataRate(RF24_250KBPS)
            self.radio.setChannel(Config.RF_CHANNEL)
            self.radio.setPayloadSize(Config.RF_PAYLOAD_SIZE)
            self.radio.setAutoAck(False)  # Disabled for compatibility
            self.radio.enableDynamicPayloads()
            self.radio.setRetries(5, 15)
            
            # Open reading pipe
            self.radio.openReadingPipe(0, Config.RF_ADDRESS)
            self.radio.startListening()
            
            self._print_configuration()
            return True
            
        except Exception as e:
            print(f"[ERROR] RF24 initialization failed: {e}")
            return False
    
    def _print_configuration(self) -> None:
        """Print radio configuration details"""
        print("\n" + "="*50)
        print("RF24 RADIO CONFIGURATION")
        print("="*50)
        print(f"CE Pin:       GPIO{Config.RF_CE_PIN}")
        print(f"CSN Pin:      SPI CE{Config.RF_CSN_PIN}")
        print(f"Channel:      {Config.RF_CHANNEL} (2.476 GHz)")
        print(f"Data Rate:    250 kbps")
        print(f"PA Level:     LOW")
        print(f"Address:      {Config.RF_ADDRESS.decode('ascii')} (0x{Config.RF_ADDRESS.hex()})")
        print(f"Payload Size: {Config.RF_PAYLOAD_SIZE} bytes")
        print(f"Auto-Ack:     Disabled")
        print("="*50 + "\n")
        
        if self.radio:
            self.radio.printDetails()
    
    def start(self) -> bool:
        """Start background receiver thread"""
        if not self.radio:
            return False
        
        self.running = True
        self.receive_thread = threading.Thread(
            target=self._receive_loop,
            daemon=True,
            name="RF24ReceiverThread"
        )
        self.receive_thread.start()
        print("[RECEIVER] Background thread started")
        return True
    
    def stop(self) -> None:
        """Stop receiver and cleanup"""
        print("\n[RECEIVER] Stopping...")
        self.running = False
        
        if self.receive_thread:
            self.receive_thread.join(timeout=2.0)
        
        if self.radio:
            self.radio.stopListening()
            print("[RECEIVER] Radio stopped")
    
    def _receive_loop(self) -> None:
        """
        Main receive loop (runs in background thread)
        Continuously polls radio for packets and parses them
        """
        consecutive_errors = 0
        print("[RECEIVER] Listening for packets...")
        
        while self.running:
            try:
                # Check for available data
                has_payload, pipe_num = self.radio.available_pipe()
                
                if has_payload:
                    payload_size = self.radio.getDynamicPayloadSize()
                    
                    if payload_size < 11:
                        consecutive_errors += 1
                        continue
                    
                    # Read and parse payload
                    payload = self.radio.read(payload_size)
                    spike = self._parse_packet(payload)
                    
                    if spike:
                        self.spike_queue.put(spike)
                        self.stats.packets_received += 1
                        consecutive_errors = 0
                        
                        # Periodic status
                        if self.stats.packets_received % 100 == 0:
                            print(f"[STATS] Received {self.stats.packets_received} packets")
                    else:
                        self.stats.parse_errors += 1
                        consecutive_errors += 1
                
                time.sleep(Config.RECEIVER_LOOP_DELAY)
                
            except Exception as e:
                consecutive_errors += 1
                print(f"[ERROR] Receive error ({consecutive_errors}/{Config.MAX_CONSECUTIVE_ERRORS}): {e}")
                
                if consecutive_errors >= Config.MAX_CONSECUTIVE_ERRORS:
                    print("[FATAL] Too many consecutive errors, stopping")
                    self.running = False
                    break
                
                time.sleep(0.1)
        
        print("[RECEIVER] Loop ended")
    
    def _parse_packet(self, payload: bytes) -> Optional[SpikeEvent]:
        """
        Parse binary packet into SpikeEvent
        
        Packet format (16 bytes):
            [0]    sensor_id (uint8)
            [1-4]  timestamp (int32)
            [5]    encoding_type (uint8)
            [6]    neuron_id (uint8)
            [7-10] polarity (float32)
            [11-15] padding
        
        Struct format: <BiBBf
        """
        try:
            # Unpack binary data
            sensor_id, timestamp, type_id, neuron_id, polarity = \
                struct.unpack('<BiBBf', bytes(payload[:11]))
            
            # Map IDs to names
            sensor_name = Config.SENSOR_NAMES[sensor_id] \
                if sensor_id < len(Config.SENSOR_NAMES) else 'unknown'
            encoding_name = Config.ENCODING_NAMES[type_id] \
                if type_id < len(Config.ENCODING_NAMES) else 'unknown'
            
            return SpikeEvent(
                sensor_id=sensor_name,
                timestamp=timestamp,
                encoding_type=encoding_name,
                neuron_id=neuron_id,
                polarity=polarity
            )
            
        except (struct.error, IndexError) as e:
            print(f"[ERROR] Packet parse error: {e}, payload: {payload.hex()}")
            return None
    
    def get_spikes(self) -> List[SpikeEvent]:
        """Get all available spikes from queue (non-blocking)"""
        spikes = []
        while not self.spike_queue.empty():
            try:
                spikes.append(self.spike_queue.get_nowait())
            except queue.Empty:
                break
        return spikes


# =============================================================================
# METRICS CALCULATOR
# =============================================================================

class SpikeMetrics:
    """Real-time spike stream analytics"""
    
    def __init__(self, window_duration: float = 1.0):
        """
        Initialize metrics calculator
        
        Args:
            window_duration: Time window for rate calculation (seconds)
        """
        self.window_duration = window_duration
        self.spike_times: Dict[str, deque] = defaultdict(deque)
        self.total_counts: Dict[str, int] = defaultdict(int)
    
    def add_spike(self, spike: SpikeEvent) -> None:
        """Add spike to metrics and update rates"""
        key = f"{spike.sensor_id}_{spike.encoding_type}"
        current_time = time.time()
        
        self.spike_times[key].append(current_time)
        self.total_counts[key] += 1
        
        # Remove old spikes outside window
        while (self.spike_times[key] and 
               current_time - self.spike_times[key][0] > self.window_duration):
            self.spike_times[key].popleft()
    
    def get_rate(self, sensor_id: str, encoding_type: str) -> int:
        """Get current spike rate in Hz"""
        key = f"{sensor_id}_{encoding_type}"
        return len(self.spike_times.get(key, []))
    
    def get_total_rate(self) -> int:
        """Get total spike rate across all sensors/encodings"""
        return sum(len(times) for times in self.spike_times.values())


# =============================================================================
# DATA LOGGER
# =============================================================================

class CSVLogger:
    """CSV file logger for spike events"""
    
    def __init__(self, filename: str):
        """Initialize CSV logger"""
        self.filename = filename
        self.file = open(filename, 'w', newline='')
        self.writer = csv.writer(self.file)
        
        # Write header
        self.writer.writerow([
            'timestamp',
            'sensor_id',
            'pico_timestamp',
            'encoding_type',
            'neuron_id',
            'polarity',
            'latency_ms'
        ])
        
        print(f"[LOGGER] Logging to {filename}")
    
    def log_spike(self, spike: SpikeEvent) -> None:
        """Write spike event to CSV"""
        self.writer.writerow([
            spike.received_time,
            spike.sensor_id,
            spike.timestamp,
            spike.encoding_type,
            spike.neuron_id,
            spike.polarity,
            spike.latency_ms
        ])
        self.file.flush()
    
    def close(self) -> None:
        """Close CSV file"""
        self.file.close()
        print(f"[LOGGER] Closed {self.filename}")


# =============================================================================
# REAL-TIME VISUALIZER
# =============================================================================

class RealtimeVisualizer:
    """Modern real-time spike visualization dashboard"""
    
    def __init__(self, receiver: RF24Receiver, metrics: SpikeMetrics, 
                 logger: Optional[CSVLogger] = None):
        """Initialize visualizer components"""
        self.receiver = receiver
        self.metrics = metrics
        self.logger = logger
        
        # Data storage
        self.spike_history: Dict[str, List[dict]] = defaultdict(list)
        self.raw_values: Dict[str, float] = {}
        self.raw_history: Dict[str, List[dict]] = defaultdict(list)
        
        # Setup visualization
        self._setup_plots()
    
    def _setup_plots(self) -> None:
        """Setup matplotlib figure and subplots"""
        plt.style.use('dark_background')
        
        self.fig, self.axes = plt.subplots(
            2, 3,
            figsize=Config.PLOT_FIGURE_SIZE
        )
        
        self.fig.suptitle(
            'Wireless Agriculture Monitoring - Neuromorphic Spike Receiver',
            fontsize=16,
            color='cyan',
            fontweight='bold'
        )
        
        # Top row: Spike raster plots
        self.encodings = ['temporal', 'rate', 'population']
        self.raster_plots = {}
        
        for i, encoding in enumerate(self.encodings):
            ax = self.axes[0, i]
            ax.set_title(f'{encoding.title()} Encoding', 
                        color='white', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time (s)', color='white')
            ax.set_ylabel('Sensor', color='white')
            ax.grid(True, alpha=0.2, linestyle='--')
            ax.set_facecolor('#1a1a1a')
            
            # Create scatter plots for each sensor
            self.raster_plots[encoding] = {}
            for j, sensor in enumerate(Config.SENSOR_NAMES):
                scatter = ax.scatter(
                    [], [],
                    c=Config.COLORS[sensor],
                    alpha=0.8,
                    s=40,
                    label=sensor,
                    edgecolors='white',
                    linewidths=0.5
                )
                self.raster_plots[encoding][sensor] = scatter
            
            ax.legend(loc='upper left', fontsize=8)
        
        # Bottom left: Metrics display
        ax_metrics = self.axes[1, 0]
        ax_metrics.set_title('System Metrics', color='cyan', 
                            fontsize=12, fontweight='bold')
        ax_metrics.axis('off')
        ax_metrics.set_facecolor('#1a1a1a')
        
        self.metrics_text = ax_metrics.text(
            0.05, 0.95, '',
            transform=ax_metrics.transAxes,
            fontsize=9,
            color='#00FF00',
            verticalalignment='top',
            fontfamily='monospace'
        )
        
        # Bottom middle: Sensor value trends
        ax_trends = self.axes[1, 1]
        ax_trends.set_title('Raw Sensor Values', color='white',
                           fontsize=12, fontweight='bold')
        ax_trends.set_xlabel('Time (s)', color='white')
        ax_trends.set_ylabel('Value', color='white')
        ax_trends.grid(True, alpha=0.2, linestyle='--')
        ax_trends.set_facecolor('#1a1a1a')
        
        self.trend_lines = {}
        for sensor in Config.SENSOR_NAMES:
            line, = ax_trends.plot(
                [], [],
                color=Config.COLORS[sensor],
                label=sensor,
                linewidth=2.5,
                alpha=0.9
            )
            self.trend_lines[sensor] = line
        
        ax_trends.legend(loc='upper left', fontsize=8)
        
        # Bottom right: Unused (turned off)
        self.axes[1, 2].axis('off')
        self.axes[1, 2].set_facecolor('#1a1a1a')
        
        plt.tight_layout()
    
    def update(self) -> None:
        """Update visualization with new data (called by animation)"""
        current_time = time.time()
        
        # Process new spikes
        new_spikes = self.receiver.get_spikes()
        self._process_spikes(new_spikes, current_time)
        
        # Update plots
        self._update_raster_plots(current_time)
        self._update_trend_plots(current_time)
        self._update_metrics_text()
        self._update_axis_limits(current_time)
    
    def _process_spikes(self, spikes: List[SpikeEvent], current_time: float) -> None:
        """Process incoming spikes and update data structures"""
        for spike in spikes:
            # Log to CSV
            if self.logger:
                self.logger.log_spike(spike)
            
            # Handle raw data separately
            if spike.is_raw_data:
                self.raw_values[spike.sensor_id] = spike.polarity
                self.raw_history[spike.sensor_id].append({
                    'time': current_time,
                    'value': spike.polarity
                })
            else:
                # Add to metrics
                self.metrics.add_spike(spike)
                
                # Add to spike history
                self.spike_history[spike.encoding_type].append({
                    'time': current_time - (time.time() - spike.received_time),
                    'sensor': spike.sensor_id,
                    'neuron': spike.neuron_id,
                    'polarity': spike.polarity
                })
        
        # Cleanup old data
        self._cleanup_old_data(current_time)
    
    def _cleanup_old_data(self, current_time: float) -> None:
        """Remove data outside retention windows"""
        # Spike history
        spike_cutoff = current_time - Config.SPIKE_HISTORY_DURATION
        for encoding in self.spike_history:
            self.spike_history[encoding] = [
                s for s in self.spike_history[encoding]
                if s['time'] > spike_cutoff
            ]
        
        # Raw value history
        raw_cutoff = current_time - Config.RAW_VALUE_HISTORY_DURATION
        for sensor in self.raw_history:
            self.raw_history[sensor] = [
                v for v in self.raw_history[sensor]
                if v['time'] > raw_cutoff
            ]
    
    def _update_raster_plots(self, current_time: float) -> None:
        """Update spike raster plots"""
        cutoff_time = current_time - Config.SPIKE_HISTORY_DURATION
        
        for encoding in self.encodings:
            if encoding not in self.raster_plots:
                continue
            
            spikes = self.spike_history.get(encoding, [])
            
            for j, sensor in enumerate(Config.SENSOR_NAMES):
                sensor_spikes = [s for s in spikes if s['sensor'] == sensor]
                
                if sensor_spikes:
                    times = [s['time'] - cutoff_time for s in sensor_spikes]
                    indices = [j] * len(sensor_spikes)
                    
                    self.raster_plots[encoding][sensor].set_offsets(
                        np.column_stack([times, indices])
                    )
    
    def _update_trend_plots(self, current_time: float) -> None:
        """Update sensor value trend lines"""
        for sensor in Config.SENSOR_NAMES:
            if sensor in self.raw_history and self.raw_history[sensor]:
                data = self.raw_history[sensor][-100:]  # Last 100 points
                times = [d['time'] - current_time for d in data]
                values = [d['value'] for d in data]
                self.trend_lines[sensor].set_data(times, values)
    
    def _update_metrics_text(self) -> None:
        """Update metrics text display"""
        lines = []
        lines.append("╔═══ RF24 WIRELESS STATUS ═══╗")
        lines.append("│ Frequency: 2.476 GHz       │")
        lines.append("│ Data Rate: 250 kbps        │")
        lines.append(f"│ Packets RX: {self.receiver.stats.packets_received:<6} │")
        lines.append("╠═══ SENSOR READINGS ═══════╣")
        
        # Sensor values
        units = {'temp': '°C', 'humid': '%', 'tds': 'ppm', 'soil': '%'}
        icons = {'temp': '🌡️ ', 'humid': '💧', 'tds': '🧪', 'soil': '🌱'}
        
        for sensor in Config.SENSOR_NAMES:
            if sensor in self.raw_values:
                value = self.raw_values[sensor]
                icon = icons.get(sensor, '')
                unit = units.get(sensor, '')
                
                if sensor == 'tds':
                    lines.append(f"│ {icon} {sensor.upper():5s}: {value:5.0f} {unit:<7}│")
                else:
                    lines.append(f"│ {icon} {sensor.capitalize():5s}: {value:5.1f} {unit:<7}│")
            else:
                lines.append(f"│ {sensor}: No data            │")
        
        lines.append("╠═══ SPIKE ACTIVITY ═════════╣")
        
        # Spike rates
        total_rate = 0
        for sensor in Config.SENSOR_NAMES:
            rate = sum(self.metrics.get_rate(sensor, enc) for enc in self.encodings)
            total_rate += rate
            if rate > 0:
                lines.append(f"│ {sensor:5s}: {rate:3.0f} Hz          │")
        
        lines.append(f"│ Total: {total_rate:3.0f} Hz           │")
        lines.append("╚═══════════════════════════╝")
        
        self.metrics_text.set_text('\n'.join(lines))
    
    def _update_axis_limits(self, current_time: float) -> None:
        """Update plot axis limits"""
        # Raster plots
        for i, encoding in enumerate(self.encodings):
            ax = self.axes[0, i]
            ax.set_xlim(0, Config.SPIKE_HISTORY_DURATION)
            ax.set_ylim(-0.5, 3.5)
            ax.set_yticks([0, 1, 2, 3])
            ax.set_yticklabels(Config.SENSOR_NAMES)
        
        # Trend plot
        self.axes[1, 1].set_xlim(-Config.RAW_VALUE_HISTORY_DURATION, 0)
        
        if any(self.raw_history.values()):
            all_values = []
            for data in self.raw_history.values():
                if data:
                    all_values.extend([d['value'] for d in data[-50:]])
            
            if all_values:
                y_min = min(all_values) - 5
                y_max = max(all_values) + 5
                self.axes[1, 1].set_ylim(y_min, y_max)
    
    def show(self) -> None:
        """Start animation and show plot window"""
        ani = animation.FuncAnimation(
            self.fig,
            lambda frame: self.update(),
            interval=Config.VISUALIZATION_UPDATE_INTERVAL,
            blit=False
        )
        plt.show()


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point"""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Wireless Agriculture Neuromorphic Receiver'
    )
    parser.add_argument('--log', default=None, 
                       help='CSV log file path')
    args = parser.parse_args()
    
    # Initialize components
    receiver = RF24Receiver()
    metrics = SpikeMetrics()
    
    # Setup logging
    log_filename = args.log or f"rf24_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    logger = CSVLogger(log_filename)
    
    # Create visualizer
    visualizer = RealtimeVisualizer(receiver, metrics, logger)
    
    # Connect to hardware
    if not receiver.connect():
        print("\n[FATAL] Failed to initialize RF24 radio")
        return
    
    if not receiver.start():
        print("[FATAL] Failed to start receiver")
        return
    
    print("\n" + "="*50)
    print("WIRELESS MONITORING ACTIVE")
    print("="*50)
    print(f"📝 Logging to: {log_filename}")
    print("📡 Waiting for Pico transmitter...")
    print("Press Ctrl+C to stop")
    print("="*50 + "\n")
    
    # Run visualization
    try:
        visualizer.show()
    except KeyboardInterrupt:
        print("\n\n[STOP] Shutting down...")
    finally:
        # Cleanup
        receiver.stop()
        logger.close()
        print("[EXIT] Shutdown complete")


if __name__ == "__main__":
    main()
