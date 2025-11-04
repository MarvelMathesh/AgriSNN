"""
Raspberry Pi Pico - Wireless Agriculture Neuromorphic Transmitter

Author: MarvelMathesh
Co-Author: Sai Siddharth
Date: 2025
"""

import machine
import utime
from machine import Pin, ADC, SPI
import dht
import math
import struct


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

class Config:
    """System-wide configuration parameters"""
    
    # Pin assignments
    DHT22_PIN = 2
    TDS_SENSOR_PIN = 26
    SOIL_SENSOR_PIN = 27
    NRF_CE_PIN = 20
    NRF_CSN_PIN = 17
    SPI_SCK_PIN = 18
    SPI_MOSI_PIN = 19
    SPI_MISO_PIN = 16
    
    # Radio configuration
    RF_CHANNEL = 76  # 2.476 GHz
    RF_ADDRESS = b'AGRIC'
    RF_PAYLOAD_SIZE = 16
    SPI_BAUDRATE = 4_000_000
    
    # Timing
    SAMPLE_INTERVAL_MS = 3000  # Sample every 3 seconds
    LOOP_DELAY_MS = 100
    MAX_CONSECUTIVE_ERRORS = 10
    
    # Sensor calibration
    TEMP_SCALE = 27.0 / 820.0
    HUMID_SCALE = 81.0 / 1792.0
    SOIL_DRY_VOLTAGE = 3.300
    SOIL_WET_VOLTAGE = 2.061
    TDS_ZERO_OFFSET = 1.501
    
    # Spike encoding thresholds
    TEMPORAL_THRESHOLD = {'temp': 1.0, 'humid': 1.0, 'tds': 5.0, 'soil': 2.0}
    POPULATION_ACTIVATION_THRESHOLD = 0.3
    POPULATION_NEURON_CENTERS = [0.0, 0.33, 0.66, 1.0]


# =============================================================================
# NRF24L01+ RADIO DRIVER
# =============================================================================

class NRF24L01Radio:
    """Modern NRF24L01+ radio driver for MicroPython"""
    
    # Register map
    class Registers:
        CONFIG = 0x00
        EN_AA = 0x01
        EN_RXADDR = 0x02
        SETUP_AW = 0x03
        SETUP_RETR = 0x04
        RF_CH = 0x05
        RF_SETUP = 0x06
        STATUS = 0x07
        TX_ADDR = 0x10
        RX_ADDR_P0 = 0x0A
        RX_PW_P0 = 0x11
        FIFO_STATUS = 0x17
        DYNPD = 0x1C
        FEATURE = 0x1D
    
    # Commands
    class Commands:
        FLUSH_TX = 0xE1
        WRITE_TX_PAYLOAD = 0xA0
    
    # Status bits
    STATUS_TX_DS = 0x20  # Data sent
    STATUS_MAX_RT = 0x10  # Max retries
    
    def __init__(self, spi, csn_pin, ce_pin):
        """Initialize radio with SPI interface"""
        self.spi = spi
        self.csn = csn_pin
        self.ce = ce_pin
        
        # Set initial pin states
        self.csn.value(1)
        self.ce.value(0)
        utime.sleep_ms(100)  # Power-on stabilization
        
        self._configure_radio()
    
    def _write_register(self, register, data):
        """Write data to radio register"""
        self.csn.value(0)
        if isinstance(data, int):
            self.spi.write(bytes([0x20 | register, data]))
        else:
            self.spi.write(bytes([0x20 | register]) + data)
        self.csn.value(1)
    
    def _read_register(self, register, length=1):
        """Read data from radio register"""
        self.csn.value(0)
        self.spi.write(bytes([register]))
        data = self.spi.read(length)
        self.csn.value(1)
        return data
    
    def _configure_radio(self):
        """Configure NRF24L01+ for TX mode with optimal settings"""
        self.ce.value(0)
        
        # Power-on reset
        self._write_register(self.Registers.CONFIG, 0x08)
        utime.sleep_ms(5)
        
        # Disable auto-acknowledgment (matches receiver)
        self._write_register(self.Registers.EN_AA, 0x00)
        
        # Enable RX pipe 0
        self._write_register(self.Registers.EN_RXADDR, 0x01)
        
        # Set 5-byte address width
        self._write_register(self.Registers.SETUP_AW, 0x03)
        
        # Set retransmission: 500µs delay, 15 retries
        self._write_register(self.Registers.SETUP_RETR, 0x1F)
        
        # Set RF channel
        self._write_register(self.Registers.RF_CH, Config.RF_CHANNEL)
        
        # Set RF power (0dBm) and data rate (250kbps)
        self._write_register(self.Registers.RF_SETUP, 0x26)
        
        # Set TX and RX addresses
        self._write_register(self.Registers.TX_ADDR, Config.RF_ADDRESS)
        self._write_register(self.Registers.RX_ADDR_P0, Config.RF_ADDRESS)
        
        # Set payload size
        self._write_register(self.Registers.RX_PW_P0, 32)
        
        # Enable dynamic payloads
        self._write_register(self.Registers.FEATURE, 0x04)
        self._write_register(self.Registers.DYNPD, 0x01)
        
        # Clear status flags
        self._write_register(self.Registers.STATUS, 0x70)
        
        # Power up in TX mode
        self._write_register(self.Registers.CONFIG, 0x0E)
        utime.sleep_ms(2)
        
        print("[RADIO] NRF24L01+ initialized (TX mode, 2.476 GHz, 250kbps)")
    
    def transmit(self, payload, timeout_ms=100):
        """
        Transmit payload packet
        
        Args:
            payload: Bytes to transmit (max 32 bytes)
            timeout_ms: Maximum wait time for transmission
            
        Returns:
            True if transmission successful, False otherwise
        """
        # Flush TX FIFO
        self.csn.value(0)
        self.spi.write(bytes([self.Commands.FLUSH_TX]))
        self.csn.value(1)
        
        # Load payload
        self.csn.value(0)
        self.spi.write(bytes([self.Commands.WRITE_TX_PAYLOAD]) + payload)
        self.csn.value(1)
        
        # Pulse CE to initiate transmission
        self.ce.value(1)
        utime.sleep_us(15)
        self.ce.value(0)
        
        # Wait for completion
        while timeout_ms > 0:
            status = self._read_register(self.Registers.STATUS)[0]
            
            if status & self.STATUS_TX_DS:
                self._write_register(self.Registers.STATUS, self.STATUS_TX_DS)
                return True
            
            if status & self.STATUS_MAX_RT:
                self._write_register(self.Registers.STATUS, self.STATUS_MAX_RT)
                return False
            
            timeout_ms -= 1
            utime.sleep_ms(1)
        
        return False


# =============================================================================
# SENSOR INTERFACE
# =============================================================================

class SensorReader:
    """Unified sensor reading interface"""
    
    def __init__(self):
        """Initialize all agriculture sensors"""
        self.dht = dht.DHT22(Pin(Config.DHT22_PIN))
        self.tds_adc = ADC(Pin(Config.TDS_SENSOR_PIN))
        self.soil_adc = ADC(Pin(Config.SOIL_SENSOR_PIN))
        print("[SENSORS] DHT22, TDS, Soil moisture initialized")
    
    def read_all(self):
        """
        Read all sensors and return calibrated values
        
        Returns:
            Dictionary with sensor readings: {'temp', 'humid', 'tds', 'soil'}
        """
        return {
            'temp': self._read_temperature(),
            'humid': self._read_humidity(),
            'tds': self._read_tds(),
            'soil': self._read_soil_moisture()
        }
    
    def _read_temperature(self):
        """Read DHT22 temperature in Celsius"""
        try:
            self.dht.measure()
            utime.sleep_ms(200)
            raw_temp = self.dht.temperature()
            calibrated = raw_temp * Config.TEMP_SCALE
            return max(-40.0, min(80.0, calibrated))
        except Exception as e:
            print(f"[SENSOR] DHT22 temperature error: {e}")
            return None
    
    def _read_humidity(self):
        """Read DHT22 humidity in percentage"""
        try:
            raw_humid = self.dht.humidity()
            calibrated = raw_humid * Config.HUMID_SCALE
            return max(0.0, min(100.0, calibrated))
        except Exception as e:
            print(f"[SENSOR] DHT22 humidity error: {e}")
            return None
    
    def _read_tds(self):
        """Read TDS (Total Dissolved Solids) in PPM"""
        try:
            raw_value = self.tds_adc.read_u16()
            voltage = (raw_value / 65535.0) * 3.3
            corrected = max(0.0, voltage - Config.TDS_ZERO_OFFSET)
            return corrected * 500.0
        except Exception as e:
            print(f"[SENSOR] TDS error: {e}")
            return None
    
    def _read_soil_moisture(self):
        """Read soil moisture in percentage"""
        try:
            raw_value = self.soil_adc.read_u16()
            voltage = (raw_value / 65535.0) * 3.3
            
            voltage_range = Config.SOIL_DRY_VOLTAGE - Config.SOIL_WET_VOLTAGE
            if voltage_range <= 0:
                return 0.0
            
            moisture = 100.0 * (Config.SOIL_DRY_VOLTAGE - voltage) / voltage_range
            return max(0.0, min(100.0, moisture))
        except Exception as e:
            print(f"[SENSOR] Soil moisture error: {e}")
            return None


# =============================================================================
# NEUROMORPHIC SPIKE ENCODER
# =============================================================================

class SpikeEncoder:
    """Neuromorphic spike encoding algorithms"""
    
    # Encoding type IDs
    ENCODING_RAW = 0
    ENCODING_TEMPORAL = 1
    ENCODING_RATE = 2
    ENCODING_POPULATION = 3
    
    # Sensor type IDs
    SENSOR_IDS = {'temp': 0, 'humid': 1, 'tds': 2, 'soil': 3}
    
    def __init__(self):
        """Initialize encoder with previous value storage"""
        self.previous_values = {
            'temp': None, 'humid': None, 'tds': None, 'soil': None
        }
    
    def create_packet(self, sensor, timestamp, encoding_type, neuron_id, polarity):
        """
        Create binary packet for wireless transmission
        
        Packet format (16 bytes):
            [0]    sensor_id (uint8)
            [1-4]  timestamp (int32)
            [5]    encoding_type (uint8)
            [6]    neuron_id (uint8)
            [7-10] polarity (float32)
            [11-15] padding (zeros)
        
        Struct format: <BiBBf (little-endian)
        """
        sensor_id = self.SENSOR_IDS.get(sensor, 0)
        
        packet = struct.pack('<BiBBf',
                            sensor_id,
                            timestamp,
                            encoding_type,
                            neuron_id,
                            polarity)
        
        # Pad to 16 bytes
        return packet + b'\x00' * (Config.RF_PAYLOAD_SIZE - len(packet))
    
    def encode_raw(self, sensor, value, timestamp):
        """Raw value encoding - direct sensor reading"""
        return self.create_packet(sensor, timestamp, self.ENCODING_RAW, 0, value)
    
    def encode_temporal(self, sensor, value, timestamp):
        """
        Temporal encoding - fires spike on significant change
        
        Returns:
            Packet if change exceeds threshold, None otherwise
        """
        prev_value = self.previous_values[sensor]
        
        if prev_value is None:
            return None
        
        change = abs(value - prev_value)
        threshold = Config.TEMPORAL_THRESHOLD.get(sensor, 1.0)
        
        if change > threshold:
            polarity = 1.0 if value > prev_value else -1.0
            return self.create_packet(sensor, timestamp, self.ENCODING_TEMPORAL, 0, polarity)
        
        return None
    
    def encode_rate(self, sensor, value, timestamp):
        """
        Rate encoding - spike probability proportional to intensity
        
        Returns:
            Packet if spike fires (probabilistic), None otherwise
        """
        # Normalize value to [0, 1]
        if sensor == 'temp':
            normalized = max(0.0, min(1.0, (value + 10.0) / 60.0))
        elif sensor == 'humid':
            normalized = max(0.0, min(1.0, value / 100.0))
        elif sensor == 'tds':
            normalized = max(0.0, min(1.0, value / 1000.0))
        else:  # soil
            normalized = max(0.0, min(1.0, value / 100.0))
        
        # Probabilistic firing (max 50% chance)
        spike_probability = normalized * 0.5
        if (timestamp % 1000) < (spike_probability * 1000):
            return self.create_packet(sensor, timestamp, self.ENCODING_RATE, 0, 1.0)
        
        return None
    
    def encode_population(self, sensor, value, timestamp):
        """
        Population encoding - distributed representation across 4 neurons
        
        Uses Gaussian tuning curves centered at different value ranges
        
        Returns:
            List of packets for active neurons
        """
        # Normalize value
        if sensor == 'temp':
            normalized = max(0.0, min(1.0, (value + 10.0) / 60.0))
        elif sensor == 'humid':
            normalized = max(0.0, min(1.0, value / 100.0))
        elif sensor == 'tds':
            normalized = max(0.0, min(1.0, value / 1000.0))
        else:  # soil
            normalized = max(0.0, min(1.0, value / 100.0))
        
        packets = []
        for neuron_id, center in enumerate(Config.POPULATION_NEURON_CENTERS):
            # Gaussian activation function
            distance = abs(normalized - center)
            activation = math.exp(-0.5 * (distance / 0.2) ** 2)
            
            if activation > Config.POPULATION_ACTIVATION_THRESHOLD:
                packet = self.create_packet(sensor, timestamp, 
                                           self.ENCODING_POPULATION,
                                           neuron_id, activation * 100.0)
                packets.append(packet)
        
        return packets
    
    def update_history(self, sensor, value):
        """Update previous value for temporal encoding"""
        self.previous_values[sensor] = value


# =============================================================================
# MAIN TRANSMITTER APPLICATION
# =============================================================================

class AgricultureTransmitter:
    """Main wireless transmitter application"""
    
    def __init__(self):
        """Initialize transmitter subsystems"""
        print("\n" + "="*50)
        print("WIRELESS AGRICULTURE NEUROMORPHIC TRANSMITTER")
        print("="*50)
        
        # Initialize SPI
        spi = SPI(0,
                 baudrate=Config.SPI_BAUDRATE,
                 polarity=0,
                 phase=0,
                 sck=Pin(Config.SPI_SCK_PIN),
                 mosi=Pin(Config.SPI_MOSI_PIN),
                 miso=Pin(Config.SPI_MISO_PIN))
        
        # Initialize subsystems
        self.radio = NRF24L01Radio(
            spi=spi,
            csn_pin=Pin(Config.NRF_CSN_PIN, Pin.OUT, value=1),
            ce_pin=Pin(Config.NRF_CE_PIN, Pin.OUT, value=0)
        )
        self.sensors = SensorReader()
        self.encoder = SpikeEncoder()
        
        # Timing
        self.start_time = utime.ticks_ms()
        self.last_sample_time = 0
        
        # Statistics
        self.packets_sent = 0
        self.packets_failed = 0
        
        print("[INIT] System ready\n")
    
    def _get_timestamp(self):
        """Get current timestamp in milliseconds since start"""
        return utime.ticks_diff(utime.ticks_ms(), self.start_time)
    
    def _transmit_packet(self, packet):
        """Transmit single packet and update statistics"""
        if self.radio.transmit(packet):
            self.packets_sent += 1
            return True
        else:
            self.packets_failed += 1
            return False
    
    def _process_sensor_reading(self, sensor, value, timestamp):
        """
        Process single sensor reading through all encoding schemes
        
        Generates and transmits:
            - 1x Raw data packet
            - 0-1x Temporal encoding packet (if change detected)
            - 0-1x Rate encoding packet (probabilistic)
            - 0-4x Population encoding packets (based on activation)
        """
        # 1. Raw data encoding
        raw_packet = self.encoder.encode_raw(sensor, value, timestamp)
        self._transmit_packet(raw_packet)
        
        # 2. Temporal encoding
        temporal_packet = self.encoder.encode_temporal(sensor, value, timestamp)
        if temporal_packet:
            self._transmit_packet(temporal_packet)
        
        # 3. Rate encoding
        rate_packet = self.encoder.encode_rate(sensor, value, timestamp)
        if rate_packet:
            self._transmit_packet(rate_packet)
        
        # 4. Population encoding
        population_packets = self.encoder.encode_population(sensor, value, timestamp)
        for packet in population_packets:
            self._transmit_packet(packet)
        
        # Update history
        self.encoder.update_history(sensor, value)
    
    def _sample_and_transmit(self):
        """Sample all sensors and transmit encoded spikes"""
        timestamp = self._get_timestamp()
        readings = self.sensors.read_all()
        
        # Display readings
        print(f"\n[{timestamp}ms] Sensor Readings:")
        for sensor, value in readings.items():
            if value is not None:
                unit = "°C" if sensor == 'temp' else "%" if sensor != 'tds' else "ppm"
                print(f"  {sensor:5s}: {value:6.1f} {unit}")
        
        # Process each sensor
        for sensor, value in readings.items():
            if value is not None:
                self._process_sensor_reading(sensor, value, timestamp)
        
        # Display statistics
        total = self.packets_sent + self.packets_failed
        success_rate = 100.0 * self.packets_sent / max(1, total)
        print(f"\n[TX] Sent: {self.packets_sent} | Failed: {self.packets_failed} | Success: {success_rate:.1f}%")
    
    def run(self):
        """Main event loop"""
        print("="*50)
        print("STARTING TRANSMISSION (2.476 GHz)")
        print("Press Ctrl+C to stop")
        print("="*50 + "\n")
        
        error_count = 0
        
        try:
            while error_count < Config.MAX_CONSECUTIVE_ERRORS:
                try:
                    current_time = utime.ticks_ms()
                    
                    # Check if sample interval elapsed
                    if utime.ticks_diff(current_time, self.last_sample_time) >= Config.SAMPLE_INTERVAL_MS:
                        self._sample_and_transmit()
                        self.last_sample_time = current_time
                        error_count = 0  # Reset on success
                    
                    utime.sleep_ms(Config.LOOP_DELAY_MS)
                    
                except Exception as e:
                    error_count += 1
                    print(f"\n[ERROR] {error_count}/{Config.MAX_CONSECUTIVE_ERRORS}: {e}")
                    utime.sleep_ms(1000)
        
        except KeyboardInterrupt:
            print("\n\n[STOP] Transmission stopped by user")
        
        finally:
            if error_count >= Config.MAX_CONSECUTIVE_ERRORS:
                print(f"\n[FATAL] Too many errors, restarting device...")
                utime.sleep(5)
                machine.reset()


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """Application entry point"""
    print("\n" + "="*50)
    print("SYSTEM STARTUP")
    print("="*50)
    print("Initializing in 3 seconds...")
    utime.sleep(3)
    
    try:
        transmitter = AgricultureTransmitter()
        transmitter.run()
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        print("Restarting in 5 seconds...")
        utime.sleep(5)
        machine.reset()


if __name__ == "__main__":
    main()