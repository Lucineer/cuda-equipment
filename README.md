# cuda-equipment

**The chandlery - shared equipment for every vessel in the fleet.**

> Every ship needs ropes, sails, and a compass before it can sail.
> Every agent needs confidence, trust, and communication before it can think.

## What It Provides

`cuda-equipment` is the foundational crate that all other fleet crates build on.

### Core Types

- **`Confidence`** - First-class uncertainty primitive (0.0 to 1.0). Fuses via harmonic mean.
- **`Tile`** / **`TileGrid`** - Spatial representation for attention, spatial reasoning, sensor data.
- **`FleetMessage`** - 11 message types for agent-to-agent communication.
- **`Agent` trait** - Interface every fleet agent implements.
- **`Fleet`** - Agent registry with register(), send(), broadcast(), ping_all().
- **`AgentBuilder`** - Fluent constructor for agents.
- **`EquipmentRegistry`** - 15 sensor types + 12 actuator types.
- **`Provenance`** - Decision lineage tracking.

### Why This Exists

In a fleet of autonomous agents, every agent needs the same basic equipment. Instead of each crate re-implementing these, cuda-equipment provides them once, correctly.

## Usage

```rust
use cuda_equipment::{Confidence, Agent, Fleet};

let mut fleet = Fleet::new();
fleet.register(my_agent);
let c1 = Confidence::new(0.8);
let c2 = Confidence::new(0.9);
let fused = c1.fuse(&c2); // harmonic mean
```

## Ecosystem Integration

This crate is a dependency of **19+ other crates** including cuda-confidence-cascade, cuda-convergence, cuda-filtration, cuda-deliberation, cuda-intent-embed, cuda-edge-runtime, cuda-resolve-agent, cuda-sensor-agent, cuda-swarm-agent, cuda-git-agent, and cuda-captain.

## Confidence Propagation

The harmonic mean formula `1/(1/a + 1/b)` is used throughout the fleet:
- `cuda-confidence` - primitive fusion
- `cuda-confidence-cascade` - cascaded fusion with gates
- `cuda-fusion` - multi-source weighted/Bayesian fusion
- `cuda-sensor-agent` - Bayesian sensor fusion
- `cuda-trust` - trust accumulation via confidence

## See Also

- [cuda-confidence](https://github.com/Lucineer/cuda-confidence) - Standalone confidence type
- [cuda-a2a](https://github.com/Lucineer/cuda-a2a) - Agent-to-agent protocol
- [cuda-actor](https://github.com/Lucineer/cuda-actor) - Actor model implementation
- [flux-runtime-c](https://github.com/Lucineer/flux-runtime-c) - C VM that executes fleet opcodes

## License

MIT OR Apache-2.0