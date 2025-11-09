# 🌾 Wireless Agriculture Neuromorphic Monitoring System (AgriSNN)

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![MicroPython](https://img.shields.io/badge/MicroPython-1.19+-green.svg)](https://micropython.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production-success.svg)]()

**A cutting-edge agricultural monitoring system combining neuromorphic computing, wireless sensor networks, and spiking neural networks for intelligent irrigation control.**

## 📋 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Hardware Requirements](#hardware-requirements)
- [Software Requirements](#software-requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Neuromorphic Spike Encoding](#neuromorphic-spike-encoding)
- [Spiking Neural Network (SNN)](#spiking-neural-network-snn)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Performance Optimization](#performance-optimization)
- [Contributing](#contributing)
- [License](#license)
- [Authors](#authors)

---

## 🎯 Overview

This project implements a **wireless agriculture monitoring system** that uses **neuromorphic computing principles** to process sensor data and make intelligent irrigation decisions. The system consists of two main components:

1. **Pico Transmitter** (`pico_transmitter.py`) - Raspberry Pi Pico-based sensor node
2. **RPi3B Receiver** (`rpi3B_receiver.py`) - Raspberry Pi 3B+ central processing unit with SNN brain

### Key Innovation: Neuromorphic Spike Encoding

Unlike traditional sampling systems, this project encodes sensor data as **spikes** (discrete events), mimicking biological neural systems. This approach offers:

- ⚡ **Energy Efficiency** - Only transmit on significant changes
- 🧠 **Bio-inspired Processing** - Natural fit for spiking neural networks
- 📡 **Reduced Bandwidth** - Event-driven communication
- 🎯 **Temporal Precision** - Capture dynamics, not just snapshots

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AGRICULTURE FIELD                         │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         RASPBERRY PI PICO TRANSMITTER                 │  │
│  │  ┌────────────┐  ┌──────────────┐  ┌─────────────┐  │  │
│  │  │  DHT22     │  │  TDS Sensor  │  │ Soil Sensor │  │  │
│  │  │ (Temp/Hum) │  │   (Water)    │  │  (Moisture) │  │  │
│  │  └─────┬──────┘  └──────┬───────┘  └──────┬──────┘  │  │
│  │        │                 │                  │         │  │
│  │        └─────────────────┴──────────────────┘         │  │
│  │                          │                            │  │
│  │                ┌─────────▼─────────┐                  │  │
│  │                │  Spike Encoder    │                  │  │
│  │                │  - Raw            │                  │  │
│  │                │  - Temporal       │                  │  │
│  │                │  - Rate           │                  │  │
│  │                │  - Population     │                  │  │
│  │                └─────────┬─────────┘                  │  │
│  │                          │                            │  │
│  │                ┌─────────▼─────────┐                  │  │
│  │                │  NRF24L01+ Radio  │                  │  │
│  │                │  2.476 GHz        │                  │  │
│  │                │  250 kbps         │                  │  │
│  │                └─────────┬─────────┘                  │  │
│  └──────────────────────────┼──────────────────────────┘  │
│                              │                             │
│                              │ Wireless (16-byte packets)  │
│                              │                             │
│  ┌──────────────────────────▼──────────────────────────┐  │
│  │       RASPBERRY PI 3B+ RECEIVER & BRAIN             │  │
│  │  ┌────────────────────────────────────────────┐     │  │
│  │  │          NRF24L01+ Radio Receiver          │     │  │
│  │  └────────────────┬───────────────────────────┘     │  │
│  │                   │                                  │  │
│  │  ┌────────────────▼───────────────────────────┐     │  │
│  │  │      Spiking Neural Network (SNN)          │     │  │
│  │  │  ┌──────────────────────────────────────┐  │     │  │
│  │  │  │  Input Layer (16 neurons)            │  │     │  │
│  │  │  │  4 sensors × 4 encoding types        │  │     │  │
│  │  │  └────────────┬─────────────────────────┘  │     │  │
│  │  │               │                             │     │  │
│  │  │  ┌────────────▼─────────────────────────┐  │     │  │
│  │  │  │  Hidden Layer (32 LIF neurons)       │  │     │  │
│  │  │  │  STDP Learning                       │  │     │  │
│  │  │  └────────────┬─────────────────────────┘  │     │  │
│  │  │               │                             │     │  │
│  │  │  ┌────────────▼─────────────────────────┐  │     │  │
│  │  │  │  Output Layer (8 decision neurons)   │  │     │  │
│  │  │  │  - Irrigation needed                 │  │     │  │
│  │  │  │  - Nutrient deficiency               │  │     │  │
│  │  │  │  - Optimal conditions                │  │     │  │
│  │  │  │  - Temperature alert                 │  │     │  │
│  │  │  │  - Humidity alert                    │  │     │  │
│  │  │  │  - Soil dry                          │  │     │  │
│  │  │  │  - Water quality low                 │  │     │  │
│  │  │  │  - System healthy                    │  │     │  │
│  │  │  └──────────────────────────────────────┘  │     │  │
│  │  └────────────────┬───────────────────────────┘     │  │
│  │                   │                                  │  │
│  │  ┌────────────────▼───────────────────────────┐     │  │
│  │  │    Irrigation Controller (GPIO Relay)      │     │  │
│  │  │    - Auto ON: < 30% soil moisture          │     │  │
│  │  │    - Auto OFF: > 70% soil moisture         │     │  │
│  │  └────────────────────────────────────────────┘     │  │
│  │                                                      │  │
│  │  ┌────────────────────────────────────────────┐     │  │
│  │  │   Real-time Visualization Dashboard        │     │  │
│  │  │   - Spike raster plots                     │     │  │
│  │  │   - Sensor value trends                    │     │  │
│  │  │   - SNN decisions                          │     │  │
│  │  │   - System metrics                         │     │  │
│  │  └────────────────────────────────────────────┘     │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

### Transmitter (Raspberry Pi Pico)

- 🌡️ **Multi-Sensor Integration**
  - DHT22: Temperature and humidity
  - TDS Sensor: Total dissolved solids (water quality)
  - Capacitive soil moisture sensor
  
- 🧮 **Neuromorphic Spike Encoding** (4 algorithms)
  - **Raw Data**: Direct sensor values
  - **Temporal**: Event-driven change detection
  - **Rate**: Intensity-proportional firing
  - **Population**: Distributed Gaussian representation

- 📡 **Wireless Communication**
  - NRF24L01+ radio at 2.476 GHz
  - 250 kbps data rate for range
  - 16-byte binary packets
  - Automatic retransmission (15 retries)

- ⚙️ **Robust Design**
  - Auto-recovery from sensor failures
  - Consecutive error handling
  - Watchdog reset on critical errors
  - Real-time statistics tracking

### Receiver (Raspberry Pi 3B+)

- 🧠 **Spiking Neural Network (SNN)**
  - 16→32→8 neuron architecture
  - Leaky Integrate-and-Fire (LIF) neurons
  - STDP learning (Spike-Timing Dependent Plasticity)
  - 8 agricultural decision outputs

- 💧 **Intelligent Irrigation Control**
  - GPIO relay control (GPIO17)
  - Hysteresis logic (30%-70% thresholds)
  - Prevents rapid cycling
  - Usage tracking and statistics

- 📊 **Real-time Visualization**
  - Spike raster plots (temporal/rate/population)
  - Sensor value trend graphs
  - System metrics dashboard
  - SNN decision display with confidence levels

- 💾 **Data Logging**
  - CSV export with timestamps
  - Latency measurements
  - Complete spike event recording

- 🔄 **Adaptive Learning**
  - 5-minute warmup period
  - Continuous STDP weight updates
  - Pattern recognition improvement over time

---

## 🛠️ Hardware Requirements

### Transmitter Node

| Component | Specification | Purpose |
|-----------|---------------|---------|
| **Microcontroller** | Raspberry Pi Pico (RP2040) | Main controller |
| **Temperature/Humidity** | DHT22 | Environmental monitoring |
| **Water Quality** | TDS Sensor (analog) | Nutrient measurement |
| **Soil Moisture** | Capacitive sensor (analog) | Irrigation monitoring |
| **Radio** | NRF24L01+ module | 2.4 GHz wireless |
| **Power** | 5V USB or battery | System power |

**Wiring (Pico):**
```
DHT22:
  - Data  → GPIO2
  - VCC   → 3.3V
  - GND   → GND

TDS Sensor:
  - Signal → GPIO26 (ADC0)
  - VCC    → 3.3V
  - GND    → GND

Soil Moisture:
  - Signal → GPIO27 (ADC1)
  - VCC    → 3.3V
  - GND    → GND

NRF24L01+:
  - CE    → GPIO20
  - CSN   → GPIO17
  - SCK   → GPIO18 (SPI0 SCK)
  - MOSI  → GPIO19 (SPI0 TX)
  - MISO  → GPIO16 (SPI0 RX)
  - VCC   → 3.3V
  - GND   → GND
```

### Receiver Station

| Component | Specification | Purpose |
|-----------|---------------|---------|
| **Computer** | Raspberry Pi 3B+ | Central processing |
| **Radio** | NRF24L01+ module | 2.4 GHz wireless |
| **Relay Module** | 5V relay (GPIO17) | Water pump control |
| **Display** | HDMI monitor | Visualization |
| **Power** | 5V 2.5A adapter | System power |

**Wiring (RPi3B+):**
```
NRF24L01+:
  - CE    → GPIO22 (Pin 15)
  - CSN   → GPIO8 (SPI CE0, Pin 24)
  - SCK   → GPIO11 (SPI SCLK, Pin 23)
  - MOSI  → GPIO10 (SPI MOSI, Pin 19)
  - MISO  → GPIO9 (SPI MISO, Pin 21)
  - VCC   → 3.3V (Pin 1)
  - GND   → GND (Pin 6)

Relay Module:
  - Signal → GPIO17 (Pin 11)
  - VCC    → 5V (Pin 2)
  - GND    → GND (Pin 9)
```

---

## 💻 Software Requirements

### Transmitter (Raspberry Pi Pico)

- **MicroPython** v1.19 or later
- **Libraries** (built-in):
  - `machine` - Hardware control
  - `utime` - Timing functions
  - `dht` - DHT22 sensor
  - `math` - Mathematical operations
  - `struct` - Binary data packing

### Receiver (Raspberry Pi 3B+)

- **Operating System**: Raspberry Pi OS (Bullseye or later)
- **Python**: 3.7+
- **Required Packages**:
  ```bash
  numpy>=1.19.0          # Numerical computing
  matplotlib>=3.3.0      # Visualization
  RPi.GPIO>=0.7.0        # GPIO control
  RF24>=1.4.0            # NRF24L01+ driver
  ```

---

## 📦 Installation

### Step 1: Transmitter Setup (Raspberry Pi Pico)

1. **Install MicroPython on Pico**:
   - Download MicroPython firmware from [micropython.org](https://micropython.org/download/rp2-pico/)
   - Hold BOOTSEL button while connecting USB
   - Copy `.uf2` file to RPI-RP2 drive

2. **Upload Transmitter Code**:
   ```bash
   # Using Thonny IDE (recommended)
   # - Open pico_transmitter.py
   # - Click "Run" → "Save" → "Raspberry Pi Pico"
   
   # Or using ampy:
   pip install adafruit-ampy
   ampy --port COM3 put pico_transmitter.py main.py
   ```

3. **Wire Hardware** according to wiring diagram above

4. **Test**: Reset Pico - should see startup messages in serial console

### Step 2: Receiver Setup (Raspberry Pi 3B+)

1. **System Dependencies**:
   ```bash
   sudo apt-get update
   sudo apt-get install -y \
       python3-dev \
       python3-pip \
       python3-rpi.gpio \
       libboost-python-dev \
       git
   ```

2. **Install RF24 Library** (critical):
   ```bash
   cd ~
   git clone https://github.com/nRF24/RF24.git
   cd RF24
   ./configure
   make
   sudo make install
   
   # Install Python bindings
   cd pyRF24
   python3 setup.py build
   sudo python3 setup.py install
   ```

3. **Install Python Packages**:
   ```bash
   pip3 install numpy matplotlib RPi.GPIO
   ```

4. **Clone Project**:
   ```bash
   cd ~/Documents
   git clone <repository-url> AgriSNN
   cd AgriSNN
   ```

5. **Enable SPI**:
   ```bash
   sudo raspi-config
   # Navigate to: Interface Options → SPI → Enable
   sudo reboot
   ```

6. **Wire Hardware** according to wiring diagram

7. **Test Radio Connection**:
   ```bash
   python3 rpi3B_receiver.py
   # Should see "RF24 RADIO CONFIGURATION" output
   ```

---

## ⚙️ Configuration

### Transmitter Configuration (`pico_transmitter.py`)

Edit the `Config` class to customize:

```python
class Config:
    # Pin assignments
    DHT22_PIN = 2
    TDS_SENSOR_PIN = 26
    SOIL_SENSOR_PIN = 27
    NRF_CE_PIN = 20
    NRF_CSN_PIN = 17
    
    # Radio configuration
    RF_CHANNEL = 76  # 2.476 GHz (must match receiver)
    RF_ADDRESS = b'AGRIC'  # 5-byte address
    
    # Timing
    SAMPLE_INTERVAL_MS = 3000  # Sample every 3 seconds
    
    # Sensor calibration (adjust based on your sensors)
    TEMP_SCALE = 27.0 / 820.0
    HUMID_SCALE = 81.0 / 1792.0
    SOIL_DRY_VOLTAGE = 3.300
    SOIL_WET_VOLTAGE = 2.061
    TDS_ZERO_OFFSET = 1.501
    
    # Spike encoding thresholds
    TEMPORAL_THRESHOLD = {
        'temp': 1.0,    # °C change to trigger spike
        'humid': 1.0,   # % change
        'tds': 5.0,     # ppm change
        'soil': 2.0     # % change
    }
```

### Receiver Configuration (`rpi3B_receiver.py`)

Edit the `Config` class:

```python
class Config:
    # Radio settings (must match transmitter)
    RF_CHANNEL = 76
    RF_ADDRESS = bytes([0x41, 0x47, 0x52, 0x49, 0x43])  # 'AGRIC'
    
    # SNN Configuration
    SNN_HIDDEN_NEURONS = 32
    SNN_OUTPUT_NEURONS = 8
    SNN_LEARNING_RATE = 0.05
    SNN_WARMUP_PERIOD = 300  # seconds
    
    # Relay Control
    RELAY_PIN = 17
    SOIL_MOISTURE_LOW_THRESHOLD = 30.0   # % - Turn ON
    SOIL_MOISTURE_HIGH_THRESHOLD = 70.0  # % - Turn OFF
    RELAY_ACTIVE_LOW = False  # Set True if relay activates on LOW
    
    # Visualization
    SPIKE_HISTORY_DURATION = 10.0  # seconds
    RAW_VALUE_HISTORY_DURATION = 60.0  # seconds
```

---

## 🚀 Usage

### Starting the System

1. **Start Transmitter First**:
   - Connect Pico via USB (with sensors wired)
   - System auto-starts (code is saved as `main.py`)
   - Watch serial output for confirmation:
     ```
     ==========================================
     WIRELESS AGRICULTURE NEUROMORPHIC TRANSMITTER
     ==========================================
     [RADIO] NRF24L01+ initialized (TX mode)
     [SENSORS] DHT22, TDS, Soil moisture initialized
     [INIT] System ready
     
     STARTING TRANSMISSION (2.476 GHz)
     ```

2. **Start Receiver**:
   ```bash
   cd ~/Documents/AgriSNN
   python3 rpi3B_receiver.py
   
   # Optional: specify log file
   python3 rpi3B_receiver.py --log my_experiment.csv
   ```

3. **Monitor Dashboard**:
   - Visualization window opens automatically
   - Shows real-time spike activity
   - Displays SNN decisions
   - Relay status visible in metrics

### Understanding the Dashboard

The visualization has 6 panels:

```
┌─────────────────┬─────────────────┬─────────────────┐
│  Temporal       │  Rate           │  Population     │
│  Encoding       │  Encoding       │  Encoding       │
│  (Spike Raster) │  (Spike Raster) │  (Spike Raster) │
├─────────────────┼─────────────────┼─────────────────┤
│  System Metrics │  Sensor Trends  │  (Reserved)     │
│  - RF24 Status  │  - Temperature  │                 │
│  - Sensors      │  - Humidity     │                 │
│  - Spike Rates  │  - TDS          │                 │
│  - SNN Brain    │  - Soil         │                 │
│  - Relay Status │                 │                 │
└─────────────────┴─────────────────┴─────────────────┘
```

**Interpreting Spike Rasters**:
- Each dot = one spike event
- Y-axis = sensor type (temp/humid/tds/soil)
- X-axis = time (10-second window)
- Color = sensor (red/green/blue/orange)

**SNN Decision Panel**:
- Shows top 3 active decisions
- Progress bar indicates confidence (0-100%)
- Learning status shows adaptation progress

**Relay Status**:
```
╠═══ RELAY CONTROL ═════════╣
│ 💧 IRRIGATION: ON         │
│ Soil: 25.3% (DRY)         │
│ Duration: 45s             │
│ Activations: 3            │
```

### Stopping the System

1. **Stop Receiver**: Press `Ctrl+C` in terminal
   - SNN state saved
   - CSV log closed
   - Relay turned OFF safely
   - GPIO cleanup performed

2. **Stop Transmitter**: Disconnect Pico USB or press `Ctrl+C` in serial console

---

## 🧬 Neuromorphic Spike Encoding

This system implements **four distinct spike encoding algorithms**, each providing unique information about sensor dynamics:

### 1. Raw Data Encoding

**Purpose**: Baseline sensor readings

**Mechanism**:
```python
def encode_raw(sensor, value, timestamp):
    # Direct value transmission
    return create_packet(sensor, timestamp, ENCODING_RAW, 0, value)
```

**Packet Format**:
- Sensor ID: 0-3 (temp/humid/tds/soil)
- Timestamp: Pico uptime (ms)
- Encoding Type: 0 (raw)
- Neuron ID: 0
- Polarity: Actual sensor value

**When Used**: Every sample (3-second interval)

**Example**:
```
Temperature: 25.3°C → Raw packet: polarity=25.3
```

### 2. Temporal Encoding

**Purpose**: Detect significant changes (event-driven)

**Mechanism**:
```python
def encode_temporal(sensor, value, timestamp):
    change = abs(value - previous_value)
    if change > threshold:
        polarity = +1.0 if value > previous else -1.0
        return create_packet(sensor, timestamp, ENCODING_TEMPORAL, 0, polarity)
    return None  # No spike if below threshold
```

**Characteristics**:
- Fires **only on significant change**
- Polarity: +1.0 (increase) or -1.0 (decrease)
- Thresholds:
  - Temperature: 1.0°C
  - Humidity: 1.0%
  - TDS: 5.0 ppm
  - Soil: 2.0%

**Example**:
```
t=0s: Temp=25.0°C → No spike (first reading)
t=3s: Temp=25.5°C → No spike (Δ=0.5°C < 1.0°C)
t=6s: Temp=26.8°C → SPIKE! (Δ=1.3°C > 1.0°C, polarity=+1.0)
```

**Advantage**: Bandwidth efficient - only transmit on events

### 3. Rate Encoding

**Purpose**: Intensity-proportional firing (probabilistic)

**Mechanism**:
```python
def encode_rate(sensor, value, timestamp):
    normalized = normalize_to_0_1(value)
    spike_probability = normalized * 0.5  # Max 50% chance
    if random() < spike_probability:
        return create_packet(sensor, timestamp, ENCODING_RATE, 0, 1.0)
    return None
```

**Characteristics**:
- Higher values → Higher firing rate
- Stochastic (probabilistic)
- Captures intensity as temporal frequency

**Example**:
```
Soil Moisture: 20% → ~10% firing probability
Soil Moisture: 80% → ~40% firing probability
```

**Advantage**: Natural for time-varying signals

### 4. Population Encoding

**Purpose**: Distributed representation (coarse coding)

**Mechanism**:
```python
def encode_population(sensor, value, timestamp):
    normalized = normalize_to_0_1(value)
    packets = []
    
    for neuron_id, center in enumerate([0.0, 0.33, 0.66, 1.0]):
        # Gaussian tuning curve
        distance = abs(normalized - center)
        activation = exp(-0.5 * (distance / 0.2)^2)
        
        if activation > 0.3:
            packets.append(create_packet(sensor, timestamp, 
                                        ENCODING_POPULATION,
                                        neuron_id, activation * 100))
    return packets
```

**Characteristics**:
- 4 neurons per sensor (16 total)
- Each neuron "tuned" to different value range:
  - Neuron 0: Low values (0-25%)
  - Neuron 1: Low-mid values (25-50%)
  - Neuron 2: Mid-high values (50-75%)
  - Neuron 3: High values (75-100%)
- Gaussian activation curves (σ=0.2)

**Example**:
```
Temperature: 15°C (normalized=0.42)
  → Neuron 1 activation: 95% (SPIKE!)
  → Neuron 2 activation: 78% (SPIKE!)
  → Neuron 0 activation: 22% (no spike)
  → Neuron 3 activation: 8% (no spike)
```

**Advantage**: Robust to noise, natural for SNN input

### Encoding Comparison

| Encoding | Spike Rate | Information | Bandwidth | Best For |
|----------|-----------|-------------|-----------|----------|
| **Raw** | Fixed (0.33 Hz) | Absolute values | High | Baseline data |
| **Temporal** | Variable (event-driven) | Changes | Low | Edge detection |
| **Rate** | Proportional to intensity | Magnitude | Medium | Intensity coding |
| **Population** | Distributed (0-4 spikes) | Coarse value | High | SNN input |

---

## 🧠 Spiking Neural Network (SNN)

The receiver implements a **3-layer Leaky Integrate-and-Fire (LIF)** spiking neural network with **STDP learning** for agricultural decision-making.

### Network Architecture

```
Input Layer (16 neurons)
  ↓
Hidden Layer (32 LIF neurons)
  ↓
Output Layer (8 decision neurons)
```

**Input Encoding**:
```
4 sensors × 4 encoding types = 16 input neurons

Neuron 0:  temp_raw_data
Neuron 1:  temp_temporal
Neuron 2:  temp_rate
Neuron 3:  temp_population
Neuron 4:  humid_raw_data
Neuron 5:  humid_temporal
...
Neuron 15: soil_population
```

**Output Decisions**:
1. **irrigation_needed** - Soil moisture critically low
2. **nutrient_deficiency** - TDS levels insufficient
3. **optimal_conditions** - All parameters in range
4. **temperature_alert** - Temperature extremes detected
5. **humidity_alert** - Humidity out of bounds
6. **soil_dry** - Soil moisture below target
7. **water_quality_low** - TDS concentration low
8. **system_healthy** - All sensors functioning normally

### Leaky Integrate-and-Fire (LIF) Neuron Model

**Dynamics**:
```python
# Membrane potential equation
V(t+1) = decay * V(t) + I_syn(t)

# Firing condition
if V(t) >= threshold:
    spike = True
    V(t) = 0  # Reset
    refractory_counter = 5  # 5ms refractory period
```

**Parameters**:
- Threshold: 1.0
- Decay rate: 0.95 (5% leak per timestep)
- Refractory period: 5 ms

**Physical Interpretation**:
- `V(t)`: Membrane potential (accumulates charge)
- `I_syn`: Synaptic input current (weighted sum of input spikes)
- `decay`: Passive leak (charge dissipation)
- `refractory`: Recovery period after spike

### STDP Learning Algorithm

**Spike-Timing Dependent Plasticity** adjusts synaptic weights based on spike timing:

```python
# Pre-synaptic spike → Post-synaptic spike (within Δt < 20ms)
if pre_fires_before_post:
    Δw = +learning_rate * pre_trace * post_spike  # Potentiation
    
# Post-synaptic spike → Pre-synaptic spike
if post_fires_before_pre:
    Δw = -learning_rate * post_trace * pre_spike  # Depression

# Weight update
w_new = clip(w_old + Δw, -1.0, 1.0)
```

**Traces** (exponential decay):
```python
trace(t) = 0.9 * trace(t-1) + spike(t)
```

**Learning Rate Schedule**:
- Warmup (0-5 min): α = 0.05 (fast adaptation)
- Normal (5+ min): α = 0.05 (continued learning)

**Biological Motivation**:
- "Neurons that fire together, wire together" (Hebbian)
- Causality detection (pre → post strengthens connection)
- Temporal credit assignment

### Training and Adaptation

**Warmup Period** (5 minutes):
- Network learns initial patterns
- Threshold lowered (0.1) for early feedback
- Statistics: "Learning basic patterns..."

**Operational Phase** (5+ minutes):
- Pattern recognition stabilizes
- Full threshold (0.3) for confident decisions
- Statistics: "Pattern recognition active"

**Decision Confidence**:
```
Confidence = activation_level × 100%

Low:      0-30%   → Uncertain
Medium:   30-60%  → Moderate
High:     60-100% → Confident
```

**Example Evolution**:
```
t=0 min:   Random weights → "Analyzing data..."
t=2 min:   "Learning basic patterns... (40%)"
t=5 min:   "Pattern recognition active (100%)"
t=10 min:  "Optimal performance (100%)" + high confidence decisions
```

### Decision-Making Logic

**Activation Calculation**:
```python
# Exponential moving average
α_fast = 0.3   # During warmup
α_slow = 0.1   # During operation

decision[i] = (1-α) * decision[i] + α * output_spike[i]
```

**Top Decisions**:
```python
active_decisions = [
    (label, activation)
    for label, activation in decisions.items()
    if activation > threshold
]
sorted_decisions = sort(active_decisions, by_activation, descending)
```

**Recommendation Generation**:
```python
if "irrigation_needed" activation > 0.5:
    → "🚨 CRITICAL: Irrigation required (78% confidence)"
    → Auto-activate relay if soil < 30%

if "optimal_conditions" activation > 0.6:
    → "✅ All parameters optimal (85% confidence)"
    → Deactivate relay if soil > 70%
```

### SNN Performance Metrics

Monitor these in the dashboard:

```
╠═══ SNN BRAIN STATUS ═════════╣
│ Learning: 100% ✓              │
│ Status: Optimal performance   │
│ Decisions (confidence):       │
│  ├─ irrigation_needed (78%)   │
│  ├─ soil_dry (65%)            │
│  └─ water_quality_low (42%)   │
```

---

## 📡 API Reference

### Transmitter Classes

#### `Config`
System configuration constants.

**Key Attributes**:
- `RF_CHANNEL`: Radio frequency channel (76 = 2.476 GHz)
- `SAMPLE_INTERVAL_MS`: Sensor sampling rate (3000 ms)
- `TEMPORAL_THRESHOLD`: Change thresholds for temporal encoding

#### `NRF24L01Radio`
NRF24L01+ radio driver.

**Methods**:
- `__init__(spi, csn_pin, ce_pin)`: Initialize radio
- `transmit(payload, timeout_ms=100)`: Transmit packet
  - **Returns**: `True` on success, `False` on failure

**Configuration**:
```python
radio = NRF24L01Radio(spi, csn, ce)
success = radio.transmit(b'\x01\x02\x03...')  # 16-byte payload
```

#### `SensorReader`
Unified sensor interface.

**Methods**:
- `read_all()`: Read all sensors
  - **Returns**: `{'temp': float, 'humid': float, 'tds': float, 'soil': float}`

**Example**:
```python
sensors = SensorReader()
readings = sensors.read_all()
print(f"Temperature: {readings['temp']}°C")
```

#### `SpikeEncoder`
Neuromorphic encoding engine.

**Methods**:
- `encode_raw(sensor, value, timestamp)`: Raw encoding
- `encode_temporal(sensor, value, timestamp)`: Temporal encoding
- `encode_rate(sensor, value, timestamp)`: Rate encoding
- `encode_population(sensor, value, timestamp)`: Population encoding
- `update_history(sensor, value)`: Update previous value

**Packet Format**:
```python
struct.pack('<BiBBf',
    sensor_id,      # uint8 (0-3)
    timestamp,      # int32 (ms)
    encoding_type,  # uint8 (0-3)
    neuron_id,      # uint8 (0-3)
    polarity        # float32
)
# Total: 11 bytes + 5 bytes padding = 16 bytes
```

#### `AgricultureTransmitter`
Main transmitter application.

**Methods**:
- `run()`: Main event loop (blocking)

**Usage**:
```python
transmitter = AgricultureTransmitter()
transmitter.run()  # Runs until Ctrl+C or fatal error
```

### Receiver Classes

#### `RF24Receiver`
NRF24L01+ receiver using RF24 library.

**Methods**:
- `connect()`: Initialize radio hardware
  - **Returns**: `True` on success
- `start()`: Start background receiver thread
- `stop()`: Stop receiver and cleanup
- `get_spikes()`: Get all received spikes
  - **Returns**: `List[SpikeEvent]`

**Example**:
```python
receiver = RF24Receiver()
if receiver.connect():
    receiver.start()
    spikes = receiver.get_spikes()
```

#### `SpikeEvent`
Container for spike data.

**Attributes**:
- `sensor_id`: str ('temp', 'humid', 'tds', 'soil')
- `timestamp`: int (Pico timestamp in ms)
- `encoding_type`: str ('raw_data', 'temporal', 'rate', 'population')
- `neuron_id`: int (0-3)
- `polarity`: float (value or ±1.0)
- `received_time`: float (receiver timestamp)
- `latency_ms`: float (transmission latency)

#### `AgricultureSNN`
Spiking neural network brain.

**Methods**:
- `process_spike(spike)`: Process incoming spike
  - **Returns**: `Dict[str, float]` (decision activations)
- `get_top_decisions(threshold=0.3)`: Get active decisions
  - **Returns**: `List[Tuple[str, float]]`
- `get_learning_progress()`: Get training progress
  - **Returns**: `(progress: float, status: str)`
- `get_recommendation()`: Generate human-readable recommendation
  - **Returns**: `str`

**Example**:
```python
snn = AgricultureSNN()
decisions = snn.process_spike(spike)

top_decisions = snn.get_top_decisions(threshold=0.3)
for decision, confidence in top_decisions:
    print(f"{decision}: {confidence:.1%}")
```

#### `IrrigationController`
GPIO relay controller.

**Methods**:
- `update(soil_moisture)`: Update relay based on moisture
  - **Returns**: `True` if state changed
- `manual_override(state)`: Manual control
- `get_status()`: Get current status
  - **Returns**: `Dict` with relay state, moisture, activations
- `cleanup()`: GPIO cleanup

**Example**:
```python
relay = IrrigationController()

# Automatic control
relay.update(25.0)  # Below 30% → Turn ON

# Manual override
relay.manual_override(True)  # Force ON
relay.manual_override(False)  # Force OFF

# Check status
status = relay.get_status()
print(f"Relay active: {status['active']}")
```

#### `RealtimeVisualizer`
Matplotlib-based dashboard.

**Methods**:
- `update()`: Update all plots (called by animation)
- `show()`: Display and run animation loop

**Example**:
```python
viz = RealtimeVisualizer(receiver, metrics, logger, snn, relay)
viz.show()  # Blocks until window closed
```

---

## 🔧 Troubleshooting

### Transmitter Issues

#### Problem: "DHT22 temperature error"
**Symptoms**: Frequent sensor read failures

**Solutions**:
1. Check wiring (especially data pin to GPIO2)
2. Add 10kΩ pull-up resistor on DHT22 data line
3. Replace DHT22 (may be faulty)
4. Increase read delay:
   ```python
   self.dht.measure()
   utime.sleep_ms(500)  # Increase from 200ms
   ```

#### Problem: Radio transmission fails (0% success rate)
**Symptoms**: `packets_failed` increases, `packets_sent = 0`

**Solutions**:
1. Check NRF24L01+ wiring (especially CE/CSN pins)
2. Verify 3.3V power (use multimeter)
3. Add 10µF capacitor across VCC/GND at NRF24 module
4. Reduce distance between transmitter/receiver
5. Test with example code:
   ```python
   # Simple test
   radio.transmit(b'\xFF' * 16)  # All 0xFF pattern
   ```

#### Problem: "Too many errors, restarting device"
**Symptoms**: System keeps resetting

**Solutions**:
1. Check all sensor connections
2. Verify power supply (needs stable 5V)
3. Increase `MAX_CONSECUTIVE_ERRORS` to 20 (temporary debug)
4. Add debug prints to identify failing sensor

### Receiver Issues

#### Problem: "RF24 library not found"
**Symptoms**: Import error on startup

**Solutions**:
```bash
# Reinstall RF24
cd ~
rm -rf RF24
git clone https://github.com/nRF24/RF24.git
cd RF24
sudo ./configure
sudo make install

# Python bindings
cd pyRF24
python3 setup.py build
sudo python3 setup.py install

# Verify installation
python3 -c "from RF24 import RF24; print('OK')"
```

#### Problem: "Waiting for Pico transmitter..." forever
**Symptoms**: No packets received

**Solutions**:
1. Verify transmitter is running (check serial output)
2. Check radio channel matches (both should be 76)
3. Check RF_ADDRESS matches:
   ```python
   # Transmitter
   RF_ADDRESS = b'AGRIC'
   
   # Receiver
   RF_ADDRESS = bytes([0x41, 0x47, 0x52, 0x49, 0x43])  # Same!
   ```
4. Test with RF24 scanner example
5. Check for interference (2.4 GHz is crowded)
6. Reduce distance to < 1 meter for testing

#### Problem: SNN always shows "Analyzing data..."
**Symptoms**: No decisions after 5+ minutes

**Solutions**:
1. Verify spikes are being received (check spike rate > 0)
2. Check input encoding map:
   ```python
   print(snn.input_map)  # Should show 16 mappings
   ```
3. Lower decision threshold temporarily:
   ```python
   top_decisions = snn.get_top_decisions(threshold=0.1)
   ```
4. Increase learning rate:
   ```python
   SNN_LEARNING_RATE = 0.1  # Default: 0.05
   ```

#### Problem: Relay not activating
**Symptoms**: Relay status shows OFF despite low soil moisture

**Solutions**:
1. Check GPIO17 wiring
2. Verify RPi.GPIO installed:
   ```bash
   python3 -c "import RPi.GPIO; print('OK')"
   ```
3. Test manual control:
   ```python
   relay.manual_override(True)  # Should hear click
   ```
4. Check relay module power (needs 5V)
5. If active-low relay, set:
   ```python
   RELAY_ACTIVE_LOW = True
   ```

#### Problem: "Permission denied" GPIO error
**Symptoms**: GPIO access fails

**Solutions**:
```bash
# Add user to gpio group
sudo usermod -a -G gpio $USER
sudo reboot

# Or run with sudo (not recommended)
sudo python3 rpi3B_receiver.py
```

### Visualization Issues

#### Problem: "No display found" or blank window
**Symptoms**: Matplotlib won't display

**Solutions**:
```bash
# Check display
echo $DISPLAY  # Should show :0 or similar

# If SSH'ing, enable X11 forwarding
ssh -X pi@192.168.1.100

# Or use VNC for remote desktop
```

#### Problem: Plots not updating
**Symptoms**: Static display, no animation

**Solutions**:
1. Close other matplotlib windows
2. Restart receiver
3. Check animation interval:
   ```python
   VISUALIZATION_UPDATE_INTERVAL = 100  # Increase if slow
   ```

---

## ⚡ Performance Optimization

### Transmitter Optimization

**Battery Life**:
```python
# Reduce sampling rate
SAMPLE_INTERVAL_MS = 5000  # 5 seconds instead of 3

# Power down between samples
import machine
machine.lightsleep(SAMPLE_INTERVAL_MS)
```

**Radio Range**:
```python
# Increase TX power (0dBm → -6dBm → -12dBm → -18dBm)
RF_SETUP = 0x26  # 0dBm (max)

# Lower data rate for better range
RF_SETUP = 0x26  # 250kbps (best range)
RF_SETUP = 0x06  # 1Mbps (medium)
RF_SETUP = 0x0E  # 2Mbps (worst range)
```

### Receiver Optimization

**CPU Usage**:
```python
# Reduce visualization update rate
VISUALIZATION_UPDATE_INTERVAL = 500  # 500ms instead of 100ms

# Simplify plots (remove trend history)
RAW_VALUE_HISTORY_DURATION = 30.0  # 30s instead of 60s
```

**Memory Usage**:
```python
# Limit spike history
SPIKE_HISTORY_DURATION = 5.0  # 5s instead of 10s

# Reduce SNN size
SNN_HIDDEN_NEURONS = 16  # 16 instead of 32
```

**SNN Learning Speed**:
```python
# Faster convergence
SNN_LEARNING_RATE = 0.1  # Higher = faster (but less stable)
SNN_WARMUP_PERIOD = 120  # 2 minutes instead of 5
```

---

## 📊 Data Analysis

### CSV Log Format

Generated log file structure:

```csv
timestamp,sensor_id,pico_timestamp,encoding_type,neuron_id,polarity,latency_ms
1699564800.123,temp,1234,raw_data,0,25.3,45.2
1699564800.156,temp,1234,temporal,0,1.0,47.8
1699564800.189,temp,1234,population,1,78.5,48.1
```

**Fields**:
- `timestamp`: Receiver Unix timestamp (seconds)
- `sensor_id`: Sensor type (temp/humid/tds/soil)
- `pico_timestamp`: Transmitter timestamp (ms since boot)
- `encoding_type`: Encoding algorithm
- `neuron_id`: Population neuron (0-3) or 0
- `polarity`: Spike value or activation
- `latency_ms`: Transmission delay

### Analysis Examples

**Load data**:
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('rf24_log_20251109_143022.csv')
df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
```

**Spike rate over time**:
```python
# Resample to 1-second bins
spike_rate = df.groupby([
    df['datetime'].dt.floor('1s'),
    'sensor_id'
]).size().unstack(fill_value=0)

spike_rate.plot(figsize=(12, 6))
plt.ylabel('Spikes per second')
plt.title('Spike Rate by Sensor')
plt.show()
```

**Raw sensor values**:
```python
# Extract raw data only
raw_df = df[df['encoding_type'] == 'raw_data']

for sensor in ['temp', 'humid', 'tds', 'soil']:
    sensor_df = raw_df[raw_df['sensor_id'] == sensor]
    plt.plot(sensor_df['datetime'], sensor_df['polarity'], label=sensor)

plt.legend()
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()
```

**Latency statistics**:
```python
print(f"Mean latency: {df['latency_ms'].mean():.1f} ms")
print(f"95th percentile: {df['latency_ms'].quantile(0.95):.1f} ms")
print(f"Max latency: {df['latency_ms'].max():.1f} ms")
```

**Encoding distribution**:
```python
encoding_counts = df['encoding_type'].value_counts()
print(encoding_counts)

# Expected ratio: raw > population > temporal > rate
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

### Code Style

- **Python**: Follow PEP 8
- **MicroPython**: Keep imports minimal
- **Comments**: Document complex algorithms
- **Type hints**: Use where appropriate (receiver only)

### Testing

Before submitting PR:

1. Test transmitter on actual Pico hardware
2. Test receiver with real radio communication
3. Verify SNN learning converges (10+ minutes)
4. Check relay control with actual hardware

### Pull Request Process

1. Fork repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push: `git push origin feature/amazing-feature`
5. Open Pull Request with description

### Feature Ideas

- [ ] MQTT integration for IoT platforms
- [ ] Mobile app for remote monitoring
- [ ] Multi-zone support (multiple transmitters)
- [ ] Advanced SNN architectures (attention mechanisms)
- [ ] Energy harvesting (solar power)
- [ ] Weather API integration
- [ ] Predictive maintenance (sensor health)
- [ ] Cloud logging (Firebase, AWS IoT)

---

## 📄 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2025 MarvelMathesh, Sai Siddharth

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 👥 Authors

**MarvelMathesh** - Lead Developer
- Architecture design
- Neuromorphic encoding implementation
- SNN development
- System integration

**Sai Siddharth** - Co-Developer
- Hardware integration
- Sensor calibration
- Testing and validation
- Documentation

---

## 📚 References

### Academic Papers

1. **Neuromorphic Computing**:
   - Pfeiffer, M., & Pfeil, T. (2018). "Deep Learning with Spiking Neurons: Opportunities and Challenges"
   - Tavanaei, A., et al. (2019). "Deep Learning in Spiking Neural Networks"

2. **Spike Encoding**:
   - Bohte, S. M., et al. (2002). "Error-backpropagation in temporally encoded networks of spiking neurons"
   - Petro, B., et al. (2020). "Selection and Optimization of Temporal Spike Encoding Methods for Spiking Neural Networks"

3. **STDP Learning**:
   - Bi, G., & Poo, M. (1998). "Synaptic Modifications in Cultured Hippocampal Neurons"
   - Song, S., et al. (2000). "Competitive Hebbian learning through spike-timing-dependent synaptic plasticity"

4. **Agricultural IoT**:
   - Farooq, M. S., et al. (2019). "A Survey on the Role of IoT in Agriculture for the Implementation of Smart Farming"
   - Pivoto, D., et al. (2018). "Scientific development of smart farming technologies and their application in Brazil"

### Libraries and Tools

- **MicroPython**: https://micropython.org/
- **RF24 Network**: https://github.com/nRF24/RF24
- **NumPy**: https://numpy.org/
- **Matplotlib**: https://matplotlib.org/
- **RPi.GPIO**: https://pypi.org/project/RPi.GPIO/

### Hardware Documentation

- **Raspberry Pi Pico**: https://www.raspberrypi.com/documentation/microcontrollers/raspberry-pi-pico.html
- **NRF24L01+**: https://www.nordicsemi.com/products/nrf24l01
- **DHT22**: https://www.sparkfun.com/datasheets/Sensors/Temperature/DHT22.pdf

---

**Happy Farming! 🌱🚜🤖**
