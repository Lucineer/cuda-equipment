# cuda-equipment

Shared equipment library — Confidence propagation, Tile grid, Agent trait, Fleet A2A messaging, Equipment registry. The pulleys and engines every vessel needs.

Part of the Cocapn build layer — compilers, assemblers, and code transformation.

## What It Does

### Key Types

- `Confidence(pub f64);` — core data structure
- `TileId(pub u64);` — core data structure
- `Tile<T>` — core data structure
- `TileGrid<T: Clone + Default>` — core data structure
- `VesselId(pub u64);` — core data structure
- `FleetMessage` — core data structure
- _and 9 more (see source)_

## Quick Start

```bash
# Clone
git clone https://github.com/Lucineer/cuda-equipment.git
cd cuda-equipment

# Build
cargo build

# Run tests
cargo test
```

## Usage

```rust
use cuda_equipment::*;

// See src/lib.rs for full API
// 0 unit tests included
```

### Available Implementations

- `Confidence` — see source for methods
- `fmt::Display for Confidence` — see source for methods
- `From` — see source for methods
- `Default for Confidence` — see source for methods
- `fmt::Display for TileId` — see source for methods
- `fmt::Display for VesselId` — see source for methods

## Testing

```bash
cargo test
```

0 unit tests covering core functionality.

## Architecture

This crate is part of the **Cocapn Fleet** — a git-native multi-agent ecosystem.

- **Category**: build
- **Language**: Rust
- **Dependencies**: See `Cargo.toml`
- **Status**: Active development

## Related Crates

- [cuda-bytecode-optimizer](https://github.com/Lucineer/cuda-bytecode-optimizer)
- [cuda-asm](https://github.com/Lucineer/cuda-asm)
- [cuda-forth](https://github.com/Lucineer/cuda-forth)

## Fleet Position

```
Casey (Captain)
├── JetsonClaw1 (Lucineer realm — hardware, low-level systems, fleet infrastructure)
├── Oracle1 (SuperInstance — lighthouse, architecture, consensus)
└── Babel (SuperInstance — multilingual scout)
```

## Contributing

This is a fleet vessel component. Fork it, improve it, push a bottle to `message-in-a-bottle/for-jetsonclaw1/`.

## License

MIT

---

*Built by JetsonClaw1 — part of the Cocapn fleet*
*See [cocapn-fleet-readme](https://github.com/Lucineer/cocapn-fleet-readme) for the full fleet roadmap*
