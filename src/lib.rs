//! # cuda-equipment
//!
//! Shared equipment library for the Lucineer AI fleet.
//! Every vessel needs engines, pulleys, and sensors — this crate is the chandlery.
//!
//! ## Core Types
//!
//! - **Confidence** — universal 0→1 certainty that propagates through computation
//! - **Tile / TileGrid** — rectangular data chunking with LRU eviction
//! - **FleetMessage** — A2A protocol (Consider/Resolve/Forfeit/Ping/Pong)
//! - **Agent trait** — deliberative agent interface
//! - **EquipmentRegistry** — sensor/actuator capability inventory
//! - **TileScheduler** — shared tile loading schedule
//!
//! ## Quick Start
//!
//! ```rust
//! use cuda_equipment::{Confidence, TileGrid, BaseAgent, FleetMessage, MessageType};
//!
//! // Confidence propagation
//! let c1 = Confidence::new(0.8);
//! let c2 = Confidence::new(0.6);
//! let combined = c1.combine(c2); // Bayesian: 1/(1/0.8 + 1/0.6) ≈ 0.343
//!
//! // Tile grid for weights, pheromones, thermal
//! let grid: TileGrid<f32> = TileGrid::new(1024, 1024, 64, 64);
//! assert_eq!(grid.total_tiles(), 256);
//!
//! // Agent with A2A messaging
//! let mut agent = BaseAgent::new(1, "scout");
//! agent.add_capability("thinking");
//! let ping = FleetMessage::new(VesselId(0), VesselId(1), MessageType::Ping);
//! let responses = agent.receive(&ping);
//! assert!(matches!(&responses[0].msg_type, MessageType::Pong));
//! ```

#![cfg_attr(not(test), allow(dead_code))]

use std::collections::HashMap;
use std::fmt;

// ============================================================
// CONFIDENCE — universal 0-1 certainty
// ============================================================

/// Universal confidence value — 0.0 (uncertain) to 1.0 (certain).
/// Every value in the fleet carries confidence. It propagates through
/// computation like a stain through cloth.
///
/// # Mathematical Properties
///
/// - `combine(a, b)` = 1/(1/a + 1/b) — independent Bayesian combination
/// - `chain(a, b)` = a * b — sequential probability
/// - `weighted(a, b, wa, wb)` = weighted average
/// - `discount(c, f)` = c * f — entropy/decay
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Confidence(pub f64);

impl Confidence {
    pub const ZERO: Self = Confidence(0.0);
    pub const SURE: Self = Confidence(1.0);
    pub const HALF: Self = Confidence(0.5);
    pub const LIKELY: Self = Confidence(0.75);
    pub const UNLIKELY: Self = Confidence(0.25);

    pub fn new(v: f64) -> Self { Confidence(v.clamp(0.0, 1.0)) }
    pub fn value(&self) -> f64 { self.0 }

    /// Bayesian combination of independent confidence sources.
    /// Formula: 1/(1/a + 1/b)
    pub fn combine(self, other: Self) -> Self {
        if self.0 <= 0.0 { return other; }
        if other.0 <= 0.0 { return self; }
        Confidence((1.0 / (1.0 / self.0 + 1.0 / other.0)).clamp(0.0, 1.0))
    }

    /// Sequential confidence: if A confirms B, how confident in both?
    pub fn chain(self, other: Self) -> Self { Confidence(self.0 * other.0) }

    /// Weighted average of two confidences.
    pub fn weighted(self, other: Self, w_self: f64, w_other: f64) -> Self {
        let total = w_self + w_other;
        if total <= 0.0 { return Confidence::ZERO; }
        Confidence((self.0 * w_self + other.0 * w_other) / total)
    }

    /// Discount confidence by a factor (entropy).
    pub fn discount(self, factor: f64) -> Self {
        Confidence(self.0 * factor.clamp(0.0, 1.0))
    }

    /// Decay over N rounds with a rate.
    pub fn decay(self, rounds: u32, rate: f64) -> Self {
        Confidence(self.0 * rate.powi(rounds as i32))
    }

    pub fn is_certain(&self) -> bool { self.0 >= 0.95 }
    pub fn is_likely(&self) -> bool { self.0 >= 0.5 }
    pub fn is_uncertain(&self) -> bool { self.0 < 0.3 }

    /// Pack into u8 (0-255).
    pub fn to_bits(&self) -> u8 { (self.0 * 255.0).round() as u8 }
    /// Unpack from u8.
    pub fn from_bits(b: u8) -> Self { Confidence(b as f64 / 255.0) }

    /// Threshold gate: if below threshold, return ZERO.
    pub fn gate(self, threshold: f64) -> Self {
        if self.0 >= threshold { self } else { Confidence::ZERO }
    }
}

impl fmt::Display for Confidence {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:.1}%", self.0 * 100.0)
    }
}
impl From<f64> for Confidence { fn from(v: f64) -> Self { Confidence::new(v) } }
impl Default for Confidence { fn default() -> Self { Confidence::HALF } }

// ============================================================
// TILE — rectangular data chunk
// ============================================================

/// Unique tile identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileId(pub u64);

impl fmt::Display for TileId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "T{}", self.0) }
}

/// A rectangular chunk of data — weights, pheromones, thermal values, FPGA BRAM.
/// Shared across weight streaming, swarm tiling, thermal simulation, and FPGA memory.
#[derive(Debug, Clone)]
pub struct Tile<T> {
    pub id: TileId,
    pub row: usize,
    pub col: usize,
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<T>,
    pub confidence: Confidence,
    pub last_accessed: u64,
}

impl<T: Clone + Default> Tile<T> {
    pub fn new(id: TileId, row: usize, col: usize, rows: usize, cols: usize) -> Self {
        Self { id, row, col, rows, cols,
            data: vec![T::default(); rows * cols],
            confidence: Confidence::SURE, last_accessed: 0,
        }
    }

    pub fn width(&self) -> usize { self.cols }
    pub fn height(&self) -> usize { self.rows }
    pub fn len(&self) -> usize { self.data.len() }
    pub fn is_empty(&self) -> bool { self.data.is_empty() }

    pub fn get(&self, r: usize, c: usize) -> Option<&T> {
        if r < self.rows && c < self.cols { Some(&self.data[r * self.cols + c]) }
        else { None }
    }

    pub fn set(&mut self, r: usize, c: usize, val: T) -> bool {
        if r < self.rows && c < self.cols {
            self.data[r * self.cols + c] = val;
            self.last_accessed = Tile::<T>::now();
            true
        } else { false }
    }

    pub fn touch(&self, other: &Tile<T>) -> bool {
        self.row < other.row + other.rows && other.row < self.row + self.rows
            && self.col < other.col + other.cols && other.col < self.col + self.cols
    }

    pub fn region(&self, r0: usize, c0: usize, r1: usize, c1: usize) -> Vec<&T> {
        let mut out = vec![];
        for r in r0..r1.min(self.rows) {
            for c in c0..c1.min(self.cols) {
                out.push(&self.data[r * self.cols + c]);
            }
        }
        out
    }

    fn now() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64)
    }
}

// ============================================================
// TILE GRID — managed tiled space
// ============================================================

/// Manages a tiled 2D space. Shared by weight streaming, FPGA memory,
/// swarm pheromone fields, thermal grids, and neural compiler.
#[derive(Debug, Clone)]
pub struct TileGrid<T: Clone + Default> {
    tiles: Vec<Tile<T>>,
    pub grid_rows: usize,
    pub grid_cols: usize,
    pub tile_rows: usize,
    pub tile_cols: usize,
}

impl<T: Clone + Default> TileGrid<T> {
    pub fn new(total_rows: usize, total_cols: usize, tile_rows: usize, tile_cols: usize) -> Self {
        let grid_rows = (total_rows + tile_rows - 1) / tile_rows;
        let grid_cols = (total_cols + tile_cols - 1) / tile_cols;
        let mut tiles = vec![];
        let mut next_id = 0u64;
        for r in 0..grid_rows {
            for c in 0..grid_cols {
                let actual_rows = tile_rows.min(total_rows - r * tile_rows);
                let actual_cols = tile_cols.min(total_cols - c * tile_cols);
                tiles.push(Tile::new(TileId(next_id), r, c, actual_rows, actual_cols));
                next_id += 1;
            }
        }
        Self { tiles, grid_rows, grid_cols, tile_rows, tile_cols }
    }

    pub fn get_tile(&self, gr: usize, gc: usize) -> Option<&Tile<T>> {
        if gr < self.grid_rows && gc < self.grid_cols {
            Some(&self.tiles[gr * self.grid_cols + gc])
        } else { None }
    }

    pub fn get_tile_mut(&mut self, gr: usize, gc: usize) -> Option<&mut Tile<T>> {
        if gr < self.grid_rows && gc < self.grid_cols {
            Some(&mut self.tiles[gr * self.grid_cols + gc])
        } else { None }
    }

    pub fn total_tiles(&self) -> usize { self.tiles.len() }
    pub fn iter(&self) -> impl Iterator<Item = &Tile<T>> { self.tiles.iter() }
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Tile<T>> { self.tiles.iter_mut() }

    /// LRU order for eviction.
    pub fn lru_order(&self) -> Vec<&Tile<T>> {
        let mut sorted: Vec<&Tile<T>> = self.tiles.iter().collect();
        sorted.sort_by_key(|t| t.last_accessed);
        sorted
    }

    /// Find tiles touching a region.
    pub fn tiles_in_region(&self, row: usize, col: usize, rows: usize, cols: usize) -> Vec<&Tile<T>> {
        let gr0 = row / self.tile_rows;
        let gc0 = col / self.tile_cols;
        let gr1 = (row + rows) / self.tile_rows + 1;
        let gc1 = (col + cols) / self.tile_cols + 1;
        let mut out = vec![];
        for gr in gr0..gr1.min(self.grid_rows) {
            for gc in gc0..gc1.min(self.grid_cols) {
                if let Some(t) = self.get_tile(gr, gc) { out.push(t); }
            }
        }
        out
    }

    /// Count tiles that have been accessed.
    pub fn accessed_count(&self) -> usize {
        self.tiles.iter().filter(|t| t.last_accessed > 0).count()
    }
}

// ============================================================
// FLEET MESSAGE — A2A protocol
// ============================================================

/// Unique vessel identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VesselId(pub u64);

impl fmt::Display for VesselId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "V{}", self.0) }
}

/// A2A message types for fleet coordination.
#[derive(Debug, Clone, PartialEq)]
pub enum MessageType {
    Consider { proposal_id: u64 },
    Resolve { proposal_id: u64, accepted: bool },
    Forfeit { proposal_id: u64, reason: String },
    CapabilityQuery,
    CapabilityResponse { capabilities: String },
    Ping,
    Pong,
    TileTransfer { tile_id: TileId, region: (usize, usize, usize, usize) },
    ConfidenceUpdate { topic: String, confidence: Confidence },
    Heartbeat,
    Shutdown,
}

/// An A2A message between fleet vessels.
#[derive(Debug, Clone)]
pub struct FleetMessage {
    pub id: u64,
    pub from: VesselId,
    pub to: VesselId,
    pub msg_type: MessageType,
    pub payload: Vec<u8>,
    pub confidence: Confidence,
    pub timestamp: u64,
    pub ttl: u8,
    pub in_reply_to: Option<u64>,
}

impl FleetMessage {
    pub fn new(from: VesselId, to: VesselId, msg_type: MessageType) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        static mut NEXT_ID: u64 = 0;
        let id = unsafe { NEXT_ID += 1; NEXT_ID };
        Self {
            id, from, to, msg_type, payload: vec![],
            confidence: Confidence::SURE,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64),
            ttl: 5, in_reply_to: None,
        }
    }

    pub fn reply(&self, msg_type: MessageType) -> Self {
        let mut r = FleetMessage::new(self.to, self.from, msg_type);
        r.in_reply_to = Some(self.id);
        r
    }

    pub fn is_expired(&self) -> bool { self.ttl == 0 }
    pub fn tick(&mut self) { self.ttl = self.ttl.saturating_sub(1); }
}

// ============================================================
// AGENT TRAIT — deliberative entity interface
// ============================================================

/// Every deliberative entity in the fleet implements Agent.
/// This is the engine that every boat has.
pub trait Agent: Send {
    fn id(&self) -> VesselId;
    fn name(&self) -> &str;
    fn receive(&mut self, msg: &FleetMessage) -> Vec<FleetMessage>;
    fn capabilities(&self) -> Vec<String>;
    fn self_confidence(&self) -> Confidence;
    fn is_healthy(&self) -> bool { self.self_confidence().is_likely() }
}

/// Minimally functional agent — the default engine.
#[derive(Debug, Clone)]
pub struct BaseAgent {
    pub id: VesselId,
    pub name: String,
    pub confidence: Confidence,
    capabilities_vec: Vec<String>,
    messages_sent: u64,
    messages_received: u64,
}

impl BaseAgent {
    pub fn new(id: u64, name: &str) -> Self {
        Self { id: VesselId(id), name: name.to_string(), confidence: Confidence::HALF,
            capabilities_vec: vec![], messages_sent: 0, messages_received: 0 }
    }

    pub fn add_capability(&mut self, cap: &str) { self.capabilities_vec.push(cap.to_string()); }

    pub fn send_consider(&self, to: VesselId, proposal_id: u64) -> FleetMessage {
        FleetMessage::new(self.id, to, MessageType::Consider { proposal_id })
    }
    pub fn send_resolve(&self, to: VesselId, proposal_id: u64, accepted: bool) -> FleetMessage {
        FleetMessage::new(self.id, to, MessageType::Resolve { proposal_id, accepted })
    }
    pub fn send_forfeit(&self, to: VesselId, proposal_id: u64, reason: &str) -> FleetMessage {
        FleetMessage::new(self.id, to, MessageType::Forfeit { proposal_id: proposal_id, reason: reason.to_string() })
    }
    pub fn send_ping(&self, to: VesselId) -> FleetMessage {
        FleetMessage::new(self.id, to, MessageType::Ping)
    }
}

impl Agent for BaseAgent {
    fn id(&self) -> VesselId { self.id }
    fn name(&self) -> &str { &self.name }

    fn receive(&mut self, msg: &FleetMessage) -> Vec<FleetMessage> {
        self.messages_received += 1;
        match &msg.msg_type {
            MessageType::Ping => { self.messages_sent += 1; vec![msg.reply(MessageType::Pong)] }
            MessageType::CapabilityQuery => {
                self.messages_sent += 1;
                vec![msg.reply(MessageType::CapabilityResponse { capabilities: self.capabilities_vec.join(",") })]
            }
            MessageType::ConfidenceUpdate { confidence, .. } => {
                self.confidence = self.confidence.combine(*confidence);
                vec![]
            }
            _ => vec![],
        }
    }

    fn capabilities(&self) -> Vec<String> { self.capabilities_vec.clone() }
    fn self_confidence(&self) -> Confidence { self.confidence }
}

// ============================================================
// EQUIPMENT REGISTRY
// ============================================================

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SensorType { Camera, Thermal, Lidar, Audio, Accelerometer, Gyroscope,
    Magnetometer, Pressure, Humidity, Light, Proximity, Touch, Gps, Rf, Chemical }

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ActuatorType { Motor, Servo, Linear, Stepper, Relay, Speaker,
    Display, Led, Valve, Pump, Heater, Cooler }

#[derive(Debug, Clone)]
pub struct Sensor {
    pub name: String,
    pub sensor_type: SensorType,
    pub resolution: usize,
    pub confidence: Confidence,
}

#[derive(Debug, Clone)]
pub struct Actuator {
    pub name: String,
    pub actuator_type: ActuatorType,
    pub max_force: f64,
    pub max_speed: f64,
}

/// What sensors and actuators a vessel has.
#[derive(Debug, Clone)]
pub struct EquipmentRegistry {
    pub vessel_id: VesselId,
    pub sensors: Vec<Sensor>,
    pub actuators: Vec<Actuator>,
    pub compute_units: usize,
    pub memory_bytes: usize,
}

impl EquipmentRegistry {
    pub fn new(vessel_id: u64) -> Self {
        Self { vessel_id: VesselId(vessel_id), sensors: vec![], actuators: vec![],
            compute_units: 1, memory_bytes: 0 }
    }

    pub fn add_sensor(mut self, name: &str, stype: SensorType, resolution: usize) -> Self {
        self.sensors.push(Sensor { name: name.to_string(), sensor_type: stype, resolution, confidence: Confidence::SURE });
        self
    }
    pub fn add_actuator(mut self, name: &str, atype: ActuatorType, force: f64, speed: f64) -> Self {
        self.actuators.push(Actuator { name: name.to_string(), actuator_type: atype, max_force: force, max_speed: speed });
        self
    }

    pub fn has_sensor(&self, stype: &SensorType) -> bool {
        self.sensors.iter().any(|s| &s.sensor_type == stype)
    }
    pub fn has_actuator(&self, atype: &ActuatorType) -> bool {
        self.actuators.iter().any(|a| &a.actuator_type == atype)
    }

    /// Generate character sheet — JSON-ready equipment summary.
    pub fn character_sheet(&self) -> serde_json::Value {
        serde_json::json!({
            "vessel_id": self.vessel_id.0,
            "sensors": self.sensors.iter().map(|s| serde_json::json!({
                "name": s.name, "type": format!("{:?}", s.sensor_type).to_lowercase(), "resolution": s.resolution
            })).collect::<Vec<_>>(),
            "actuators": self.actuators.iter().map(|a| serde_json::json!({
                "name": a.name, "type": format!("{:?}", a.actuator_type).to_lowercase(),
                "max_force": a.max_force, "max_speed": a.max_speed
            })).collect::<Vec<_>>(),
            "compute_units": self.compute_units,
            "memory_bytes": self.memory_bytes,
        })
    }
}

// ============================================================
// TILE SCHEDULER
// ============================================================

#[derive(Debug, Clone)]
pub struct ScheduledTile {
    pub tile_id: TileId,
    pub layer: usize,
    pub start_cycle: u64,
    pub end_cycle: u64,
    pub bram_slot: usize,
}

/// Shared tile scheduling for weight streaming, FPGA, swarm.
pub struct TileScheduler {
    pub max_concurrent: usize,
    pub bandwidth_bytes_per_cycle: f64,
    pub latency_cycles: u64,
}

impl TileScheduler {
    pub fn new(max_concurrent: usize) -> Self {
        Self { max_concurrent, bandwidth_bytes_per_cycle: 32.0, latency_cycles: 10 }
    }

    pub fn load_time_cycles(&self, tile_bytes: usize) -> u64 {
        (tile_bytes as f64 / self.bandwidth_bytes_per_cycle).ceil() as u64 + self.latency_cycles
    }

    pub fn schedule_layer<T: Clone + Default>(
        &self, grid: &TileGrid<T>, layer: usize, bram_slots: usize
    ) -> Vec<ScheduledTile> {
        let mut schedule = vec![];
        let mut cycle = 0u64;
        let mut active = 0usize;
        for tile in grid.iter() {
            if active >= bram_slots {
                cycle += self.load_time_cycles(4096);
                active = 0;
            }
            let load = self.load_time_cycles(tile.len() * 4);
            schedule.push(ScheduledTile { tile_id: tile.id, layer, start_cycle: cycle, end_cycle: cycle + load, bram_slot: active });
            cycle += 2;
            active += 1;
        }
        schedule
    }
}

// ============================================================
// PROVENANCE — value origin tracking
// ============================================================

#[derive(Debug, Clone)]
pub struct Provenance {
    pub source_agent: VesselId,
    pub operation: String,
    pub inputs: Vec<TileId>,
    pub output: Option<TileId>,
    pub confidence: Confidence,
    pub timestamp: u64,
}

// ============================================================
// AGENT BUILDER — fluent API for constructing agents
// ============================================================

/// Fluent builder for fleet agents.
pub struct AgentBuilder {
    id: u64,
    name: String,
    confidence: Confidence,
    capabilities: Vec<String>,
}

impl AgentBuilder {
    pub fn new(id: u64, name: &str) -> Self {
        Self { id, name: name.to_string(), confidence: Confidence::HALF, capabilities: vec![] }
    }
    pub fn confidence(mut self, c: Confidence) -> Self { self.confidence = c; self }
    pub fn capability(mut self, cap: &str) -> Self { self.capabilities.push(cap.to_string()); self }
    pub fn capabilities(mut self, caps: &[&str]) -> Self { self.capabilities.extend(caps.iter().map(|s| s.to_string())); self }
    pub fn build(self) -> BaseAgent {
        let mut agent = BaseAgent::new(self.id, &self.name);
        agent.confidence = self.confidence;
        for c in &self.capabilities { agent.add_capability(c); }
        agent
    }
}

// ============================================================
// FLEET — collection of agents
// ============================================================

/// A fleet of vessels that can coordinate via A2A messaging.
pub struct Fleet {
    agents: HashMap<u64, Box<dyn Agent>>,
}

impl Fleet {
    pub fn new() -> Self { Self { agents: HashMap::new() } }

    pub fn register(&mut self, agent: Box<dyn Agent>) {
        let id = agent.id().0;
        self.agents.insert(id, agent);
    }

    pub fn send(&mut self, msg: &FleetMessage) -> Vec<FleetMessage> {
        let target_id = msg.to.0;
        let mut responses = vec![];
        if let Some(agent) = self.agents.get_mut(&target_id) {
            responses = agent.receive(msg);
        }
        responses
    }

    pub fn broadcast(&mut self, from: VesselId, msg_type: MessageType) -> Vec<FleetMessage> {
        let mut all_responses = vec![];
        for (&id, agent) in &mut self.agents {
            if id != from.0 {
                let msg = FleetMessage::new(from, VesselId(id), msg_type.clone());
                all_responses.extend(agent.receive(&msg));
            }
        }
        all_responses
    }

    pub fn ping_all(&mut self, from: VesselId) -> Vec<(u64, bool)> {
        let mut results = vec![];
        for (&id, agent) in &mut self.agents {
            if id != from.0 {
                let msg = FleetMessage::new(from, VesselId(id), MessageType::Ping);
                let responses = agent.receive(&msg);
                let alive = responses.iter().any(|r| matches!(r.msg_type, MessageType::Pong));
                results.push((id, alive));
            }
        }
        results
    }

    pub fn agent(&self, id: u64) -> Option<&dyn Agent> {
        self.agents.get(&id).map(|a| a.as_ref())
    }

    pub fn agent_count(&self) -> usize { self.agents.len() }

    pub fn healthy_count(&self) -> usize {
        self.agents.values().filter(|a| a.is_healthy()).count()
    }
}

impl Default for Fleet { fn default() -> Self { Self::new() } }

// Re-export serde_json for character_sheet
pub use serde_json;
