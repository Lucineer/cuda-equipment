//! CUDA-EQUIPMENT — Shared equipment library for the Lucineer fleet
//! 
//! Every vessel in the fleet shares common equipment:
//! - Confidence propagation (0-1 certainty on every value)
//! - Payload (JSON-first values with provenance)
//! - Agent trait (deliberative thinking capability)
//! - Tiling (weight tiles, FPGA tiles, swarm tiles)
//! - Fleet coordination (A2A messaging)
//!
//! This crate IS the equipment shelf. Every boat needs pulleys.

/// Universal confidence value — 0.0 (uncertain) to 1.0 (certain)
/// Every value in the fleet carries confidence. It propagates through
/// computation like a stain through cloth.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Confidence(pub f64);

impl Confidence {
    pub const ZERO: Self = Confidence(0.0);
    pub const SURE: Self = Confidence(1.0);
    pub const HALF: Self = Confidence(0.5);
    pub const LIKELY: Self = Confidence(0.75);
    pub const UNLIKELY: Self = Confidence(0.25);
    
    pub fn new(v: f64) -> Self {
        Confidence(v.clamp(0.0, 1.0))
    }
    
    pub fn value(&self) -> f64 { self.0 }
    
    /// Bayesian combination of independent confidences
    pub fn combine(self, other: Self) -> Self {
        if self.0 <= 0.0 { return other; }
        if other.0 <= 0.0 { return self; }
        // Harmonic mean formula: 1/(1/a + 1/b)
        let combined = 1.0 / (1.0 / self.0 + 1.0 / other.0);
        Confidence(combined.clamp(0.0, 1.0))
    }
    
    /// Sequential confidence: if A confirms B, how confident are we in both?
    pub fn chain(self, other: Self) -> Self {
        Confidence(self.0 * other.0)
    }
    
    /// Weighted average
    pub fn weighted(self, other: Self, w_self: f64, w_other: f64) -> Self {
        let total = w_self + w_other;
        if total <= 0.0 { return Confidence::ZERO; }
        Confidence((self.0 * w_self + other.0 * w_other) / total)
    }
    
    /// Discount confidence by a factor (entropy)
    pub fn discount(self, factor: f64) -> Self {
        Confidence(self.0 * factor.clamp(0.0, 1.0))
    }
    
    pub fn is_certain(&self) -> bool { self.0 >= 0.95 }
    pub fn is_likely(&self) -> bool { self.0 >= 0.5 }
    pub fn is_uncertain(&self) -> bool { self.0 < 0.3 }
    
    pub fn to_bits(&self) -> u8 {
        (self.0 * 255.0).round() as u8
    }
    
    pub fn from_bits(b: u8) -> Self {
        Confidence(b as f64 / 255.0)
    }
}

impl std::fmt::Display for Confidence {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:.1}%", self.0 * 100.0)
    }
}

impl From<f64> for Confidence {
    fn from(v: f64) -> Self { Confidence::new(v) }
}

/// A tile — a rectangular chunk of data shared across weight streaming,
/// FPGA memory, swarm pheromone fields, and thermal grids.
#[derive(Debug, Clone)]
pub struct Tile<T> {
    pub row: usize,
    pub col: usize,
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<T>,
    pub confidence: Confidence,
    pub last_accessed: u64,
    pub id: TileId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileId(pub u64);

impl<T: Clone + Default> Tile<T> {
    pub fn new(id: TileId, row: usize, col: usize, rows: usize, cols: usize) -> Self {
        Self {
            row, col, rows, cols,
            data: vec![T::default(); rows * cols],
            confidence: Confidence::SURE,
            last_accessed: 0, id,
        }
    }
    
    pub fn width(&self) -> usize { self.cols }
    pub fn height(&self) -> usize { self.rows }
    pub fn len(&self) -> usize { self.data.len() }
    pub fn is_empty(&self) -> bool { self.data.is_empty() }
    
    pub fn get(&self, r: usize, c: usize) -> Option<&T> {
        if r < self.rows && c < self.cols {
            Some(&self.data[r * self.cols + c])
        } else { None }
    }
    
    pub fn set(&mut self, r: usize, c: usize, val: T) -> bool {
        if r < self.rows && c < self.cols {
            self.data[r * self.cols + c] = val;
            self.last_accessed = Self::now();
            true
        } else { false }
    }
    
    pub fn touches(&self, other: &Tile<T>) -> bool {
        self.row < other.row + other.rows && other.row < self.row + self.rows
            && self.col < other.col + other.cols && other.col < self.col + self.cols
    }
    
    fn now() -> u64 { 0 } // Override in impl with real clock
}

/// Tile grid — manages a tiled space (weights, pheromones, thermal, etc.)
#[derive(Debug, Clone)]
pub struct TileGrid<T: Clone + Default> {
    tiles: Vec<Tile<T>>,
    pub grid_rows: usize,
    pub grid_cols: usize,
    pub tile_rows: usize,
    pub tile_cols: usize,
    next_id: u64,
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
                let mut tile = Tile::new(TileId(next_id), r, c, actual_rows, actual_cols);
                tile.data = vec![T::default(); actual_rows * actual_cols];
                tiles.push(tile);
                next_id += 1;
            }
        }
        
        Self { tiles, grid_rows, grid_cols, tile_rows, tile_cols, next_id }
    }
    
    pub fn get_tile(&self, grid_row: usize, grid_col: usize) -> Option<&Tile<T>> {
        if grid_row < self.grid_rows && grid_col < self.grid_cols {
            Some(&self.tiles[grid_row * self.grid_cols + grid_col])
        } else { None }
    }
    
    pub fn get_tile_mut(&mut self, grid_row: usize, grid_col: usize) -> Option<&mut Tile<T>> {
        if grid_row < self.grid_rows && grid_col < self.grid_cols {
            Some(&mut self.tiles[grid_row * self.grid_cols + grid_col])
        } else { None }
    }
    
    pub fn total_tiles(&self) -> usize { self.tiles.len() }
    
    /// LRU order for eviction
    pub fn lru_order(&self) -> Vec<&Tile<T>> {
        let mut sorted: Vec<&Tile<T>> = self.tiles.iter().collect();
        sorted.sort_by_key(|t| t.last_accessed);
        sorted
    }
    
    /// Find tiles touching a region
    pub fn tiles_in_region(&self, row: usize, col: usize, rows: usize, cols: usize) -> Vec<&Tile<T>> {
        let gr_start = row / self.tile_rows;
        let gc_start = col / self.tile_cols;
        let gr_end = (row + rows) / self.tile_rows + 1;
        let gc_end = (col + cols) / self.tile_cols + 1;
        
        let mut result = vec![];
        for gr in gr_start..gr_end.min(self.grid_rows) {
            for gc in gc_start..gc_end.min(self.grid_cols) {
                if let Some(t) = self.get_tile(gr, gc) { result.push(t); }
            }
        }
        result
    }
}

/// A fleet message — A2A communication between vessels
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VesselId(pub u64);

#[derive(Debug, Clone, PartialEq)]
pub enum MessageType {
    /// Ask another vessel to consider a proposal
    Consider { proposal_id: u64 },
    /// Resolve — commit to a decision
    Resolve { proposal_id: u64, outcome: bool },
    /// Forfeit — give up on a proposal
    Forfeit { proposal_id: u64, reason: String },
    /// Request capability info
    CapabilityQuery,
    /// Respond with capability info
    CapabilityResponse { capabilities: String },
    /// Ping for health check
    Ping,
    /// Pong response
    Pong,
    /// Stream tile data to another vessel
    TileTransfer { tile_id: TileId, region: (usize, usize, usize, usize) },
    /// Confidence broadcast
    ConfidenceUpdate { topic: String, confidence: Confidence },
}

impl FleetMessage {
    pub fn new(from: VesselId, to: VesselId, msg_type: MessageType) -> Self {
        Self {
            id: 0, from, to, msg_type,
            payload: vec![], confidence: Confidence::SURE,
            timestamp: 0, ttl: 5, in_reply_to: None,
        }
    }
    
    pub fn reply_to(&self, msg_type: MessageType) -> Self {
        let mut reply = FleetMessage::new(self.to, self.from, msg_type);
        reply.in_reply_to = Some(self.id);
        reply
    }
    
    pub fn is_expired(&self) -> bool { self.ttl == 0 }
    pub fn decrement_ttl(&mut self) { self.ttl = self.ttl.saturating_sub(1); }
}

/// Agent trait — any deliberative entity in the fleet
/// This is the "engine" that every boat has.
pub trait Agent {
    /// Unique identifier
    fn id(&self) -> VesselId;
    
    /// Name for debugging/logging
    fn name(&self) -> &str;
    
    /// Receive a fleet message and produce response(s)
    fn receive(&mut self, msg: &FleetMessage) -> Vec<FleetMessage>;
    
    /// What this agent can do
    fn capabilities(&self) -> Vec<String>;
    
    /// Current confidence in own state
    fn self_confidence(&self) -> Confidence;
    
    /// Health check — should this agent be considered alive?
    fn is_healthy(&self) -> bool { self.self_confidence().is_likely() }
}

/// Base agent implementation — minimally functional agent
pub struct BaseAgent {
    pub id: VesselId,
    pub name: String,
    pub confidence: Confidence,
    pub capabilities: Vec<String>,
    pub inbox: Vec<FleetMessage>,
    pub messages_sent: u64,
    pub messages_received: u64,
}

impl BaseAgent {
    pub fn new(id: u64, name: &str) -> Self {
        Self {
            id: VesselId(id), name: name.to_string(),
            confidence: Confidence::HALF,
            capabilities: vec![],
            inbox: vec![], messages_sent: 0, messages_received: 0,
        }
    }
    
    pub fn with_capabilities(mut self, caps: Vec<&str>) -> Self {
        self.capabilities = caps.iter().map(|s| s.to_string()).collect();
        self
    }
}

impl Agent for BaseAgent {
    fn id(&self) -> VesselId { self.id }
    fn name(&self) -> &str { &self.name }
    
    fn receive(&mut self, msg: &FleetMessage) -> Vec<FleetMessage> {
        self.messages_received += 1;
        match &msg.msg_type {
            MessageType::Ping => {
                let pong = msg.reply_to(MessageType::Pong);
                self.messages_sent += 1;
                vec![pong]
            }
            MessageType::CapabilityQuery => {
                let caps = self.capabilities.join(",");
                let resp = msg.reply_to(MessageType::CapabilityResponse { capabilities: caps });
                self.messages_sent += 1;
                vec![resp]
            }
            MessageType::ConfidenceUpdate { topic, confidence } => {
                self.confidence = self.confidence.combine(*confidence);
                vec![]
            }
            _ => vec![],
        }
    }
    
    fn capabilities(&self) -> Vec<String> { self.capabilities.clone() }
    fn self_confidence(&self) -> Confidence { self.confidence }
}

/// Tile scheduler — shared scheduling logic for weight streaming, FPGA, swarm
pub struct TileScheduler {
    pub max_concurrent: usize,
    pub bandwidth_bytes_per_cycle: f64,
    pub latency_cycles: u64,
}

impl TileScheduler {
    pub fn new(max_concurrent: usize) -> Self {
        Self { max_concurrent, bandwidth_bytes_per_cycle: 32.0, latency_cycles: 10 }
    }
    
    /// Calculate load time for a tile
    pub fn load_time_cycles(&self, tile_bytes: usize) -> u64 {
        let transfer = (tile_bytes as f64 / self.bandwidth_bytes_per_cycle).ceil() as u64;
        transfer + self.latency_cycles
    }
    
    /// Schedule tiles for a layer, respecting BRAM constraints
    pub fn schedule_layer<T: Clone + Default>(
        &self, grid: &TileGrid<T>, layer: usize, bram_slots: usize
    ) -> Vec<ScheduledTile> {
        let mut schedule = vec![];
        let mut current_cycle = 0u64;
        let mut active_slots = 0usize;
        
        for tile in &grid.tiles {
            if active_slots >= bram_slots {
                current_cycle += self.load_time_cycles(4096);
                active_slots = 0;
            }
            schedule.push(ScheduledTile {
                tile_id: tile.id, layer,
                start_cycle: current_cycle,
                end_cycle: current_cycle + self.load_time_cycles(tile.data.len()),
                bram_slot: active_slots,
            });
            current_cycle += 2; // Overlap
            active_slots += 1;
        }
        
        schedule
    }
}

#[derive(Debug, Clone)]
pub struct ScheduledTile {
    pub tile_id: TileId,
    pub layer: usize,
    pub start_cycle: u64,
    pub end_cycle: u64,
    pub bram_slot: usize,
}

/// Provenance — tracks where a value came from
#[derive(Debug, Clone)]
pub struct Provenance {
    pub source_agent: VesselId,
    pub operation: String,
    pub inputs: Vec<TileId>,
    pub output: Option<TileId>,
    pub confidence: Confidence,
    pub timestamp: u64,
}

/// Equipment registry — what sensors/capabilities a vessel has
#[derive(Debug, Clone)]
pub struct EquipmentRegistry {
    pub vessel_id: VesselId,
    pub sensors: Vec<Sensor>,
    pub actuators: Vec<Actuator>,
    pub compute_units: usize,
    pub memory_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct Sensor {
    pub name: String,
    pub sensor_type: SensorType,
    pub resolution: usize,
    pub confidence: Confidence,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SensorType {
    Camera,
    Thermal,
    Lidar,
    Audio,
    Accelerometer,
    Gyroscope,
    Magnetometer,
    Pressure,
    Humidity,
    Light,
    Proximity,
    Touch,
    Gps,
    Rf,
    Chemical,
}

#[derive(Debug, Clone)]
pub struct Actuator {
    pub name: String,
    pub actuator_type: ActuatorType,
    pub max_force: f64,
    pub max_speed: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ActuatorType {
    Motor,
    Servo,
    Linear,
    Stepper,
    Relay,
    Speaker,
    Display,
    Led,
    Valve,
    Pump,
    Heater,
    Cooler,
}

impl EquipmentRegistry {
    pub fn new(vessel_id: u64) -> Self {
        Self {
            vessel_id: VesselId(vessel_id),
            sensors: vec![], actuators: vec![],
            compute_units: 1, memory_bytes: 0,
        }
    }
    
    pub fn add_sensor(mut self, name: &str, stype: SensorType, resolution: usize) -> Self {
        self.sensors.push(Sensor { name: name.to_string(), sensor_type: stype, resolution, confidence: Confidence::SURE });
        self
    }
    
    pub fn add_actuator(mut self, name: &str, atype: ActuatorType, force: f64, speed: f64) -> Self {
        self.actuators.push(Actuator { name: name.to_string(), actuator_type: atype, max_force: force, max_speed: speed });
        self
    }
    
    pub fn has_sensor_type(&self, stype: &SensorType) -> bool {
        self.sensors.iter().any(|s| &s.sensor_type == stype)
    }
    
    pub fn has_actuator_type(&self, atype: &ActuatorType) -> bool {
        self.actuators.iter().any(|a| &a.actuator_type == atype)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confidence_creation() {
        let c = Confidence::new(0.75);
        assert_eq!(c.value(), 0.75);
        assert!(c.is_likely());
        assert!(!c.is_certain());
    }

    #[test]
    fn test_confidence_clamp() {
        assert_eq!(Confidence::new(-0.5).value(), 0.0);
        assert_eq!(Confidence::new(1.5).value(), 1.0);
    }

    #[test]
    fn test_confidence_combine() {
        let a = Confidence::new(0.5);
        let b = Confidence::new(0.5);
        let combined = a.combine(b);
        assert!((combined.value() - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_confidence_chain() {
        let a = Confidence::new(0.8);
        let b = Confidence::new(0.7);
        assert!((a.chain(b).value() - 0.56).abs() < 0.01);
    }

    #[test]
    fn test_confidence_weighted() {
        let a = Confidence::new(0.9);
        let b = Confidence::new(0.3);
        let w = a.weighted(b, 2.0, 1.0);
        assert!((w.value() - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_confidence_zero_propagation() {
        assert_eq!(Confidence::ZERO.combine(Confidence::HALF), Confidence::HALF);
        assert_eq!(Confidence::HALF.combine(Confidence::ZERO), Confidence::HALF);
    }

    #[test]
    fn test_confidence_bits_roundtrip() {
        let c = Confidence::new(0.75);
        let bits = c.to_bits();
        let back = Confidence::from_bits(bits);
        assert!((c.value() - back.value()).abs() < 0.01);
    }

    #[test]
    fn test_tile_operations() {
        let mut tile: Tile<f64> = Tile::new(TileId(0), 0, 0, 4, 4);
        assert_eq!(tile.len(), 16);
        assert!(tile.set(2, 3, 42.0));
        assert_eq!(*tile.get(2, 3).unwrap(), 42.0);
        assert!(tile.get(4, 0).is_none()); // out of bounds
    }

    #[test]
    fn test_tile_touching() {
        let t1: Tile<f64> = Tile::new(TileId(0), 0, 0, 4, 4);
        let t2: Tile<f64> = Tile::new(TileId(1), 3, 3, 4, 4);
        let t3: Tile<f64> = Tile::new(TileId(2), 8, 8, 4, 4);
        assert!(t1.touches(&t2));  // overlap
        assert!(!t1.touches(&t3)); // no overlap
    }

    #[test]
    fn test_tile_grid() {
        let grid: TileGrid<f32> = TileGrid::new(100, 100, 32, 32);
        assert_eq!(grid.grid_rows, 4);  // ceil(100/32)
        assert_eq!(grid.grid_cols, 4);
        assert_eq!(grid.total_tiles(), 16);
        assert!(grid.get_tile(0, 0).is_some());
        assert!(grid.get_tile(5, 5).is_none());
    }

    #[test]
    fn test_tile_grid_lru() {
        let mut grid: TileGrid<f32> = TileGrid::new(64, 64, 32, 32);
        // Access one tile
        if let Some(t) = grid.get_tile_mut(1, 1) {
            t.last_accessed = 999;
        }
        let lru = grid.lru_order();
        assert_eq!(lru[0].id, TileId(0)); // First tile should be least recently used
    }

    #[test]
    fn test_tile_grid_region() {
        let grid: TileGrid<f32> = TileGrid::new(100, 100, 32, 32);
        let tiles = grid.tiles_in_region(40, 40, 30, 30);
        assert!(tiles.len() >= 4); // Should span at least 4 tiles
    }

    #[test]
    fn test_base_agent_ping() {
        let mut agent = BaseAgent::new(1, "test");
        let ping = FleetMessage::new(VesselId(0), VesselId(1), MessageType::Ping);
        let responses = agent.receive(&ping);
        assert_eq!(responses.len(), 1);
        assert!(matches!(responses[0].msg_type, MessageType::Pong));
    }

    #[test]
    fn test_base_agent_capabilities() {
        let agent = BaseAgent::new(1, "test").with_capabilities(vec!["thinking", "sensing"]);
        assert_eq!(agent.capabilities().len(), 2);
        
        let query = FleetMessage::new(VesselId(0), VesselId(1), MessageType::CapabilityQuery);
        let mut agent = agent;
        let responses = agent.receive(&query);
        assert_eq!(responses.len(), 1);
        if let MessageType::CapabilityResponse { capabilities } = &responses[0].msg_type {
            assert!(capabilities.contains("thinking"));
        }
    }

    #[test]
    fn test_fleet_message_reply() {
        let msg = FleetMessage::new(VesselId(1), VesselId(2), MessageType::Ping);
        let reply = msg.reply_to(MessageType::Pong);
        assert_eq!(reply.from, VesselId(2));
        assert_eq!(reply.to, VesselId(1));
    }

    #[test]
    fn test_message_ttl() {
        let mut msg = FleetMessage::new(VesselId(1), VesselId(2), MessageType::Ping);
        assert!(!msg.is_expired());
        msg.ttl = 0;
        assert!(msg.is_expired());
    }

    #[test]
    fn test_equipment_registry() {
        let registry = EquipmentRegistry::new(1)
            .add_sensor("front_camera", SensorType::Camera, 1920)
            .add_sensor("thermometer", SensorType::Thermal, 12)
            .add_actuator("left_motor", ActuatorType::Motor, 10.0, 1.5);
        
        assert!(registry.has_sensor_type(&SensorType::Camera));
        assert!(registry.has_sensor_type(&SensorType::Thermal));
        assert!(!registry.has_sensor_type(&SensorType::Lidar));
        assert!(registry.has_actuator_type(&ActuatorType::Motor));
        assert!(!registry.has_actuator_type(&ActuatorType::Servo));
    }

    #[test]
    fn test_tile_scheduler() {
        let scheduler = TileScheduler::new(4);
        let grid: TileGrid<f32> = TileGrid::new(128, 128, 32, 32);
        let schedule = scheduler.schedule_layer(&grid, 0, 2);
        assert!(!schedule.is_empty());
        // First tile should start at cycle 0
        assert_eq!(schedule[0].start_cycle, 0);
    }

    #[test]
    fn test_consider_resolve_protocol() {
        let mut agent = BaseAgent::new(1, "agent");
        let consider = FleetMessage::new(VesselId(2), VesselId(1),
            MessageType::Consider { proposal_id: 42 });
        let responses = agent.receive(&consider);
        assert_eq!(responses.len(), 0); // BaseAgent doesn't handle Consider
        assert_eq!(agent.messages_received, 1);
    }

    #[test]
    fn test_confidence_update_message() {
        let mut agent = BaseAgent::new(1, "agent");
        let update = FleetMessage::new(VesselId(2), VesselId(1),
            MessageType::ConfidenceUpdate { topic: "weather".to_string(), confidence: Confidence::LIKELY });
        agent.receive(&update);
        assert!((agent.self_confidence().value() - 0.625).abs() < 0.01); // combine(0.5, 0.75)
    }
}
