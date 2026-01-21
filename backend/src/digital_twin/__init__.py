"""
Digital Twin Module

Core components for Living Digital Twin functionality:
- DigitalTwinEngine: Central orchestrator
- TwinState: Lifecycle management
- TwinAlert: Alert generation
- TwinSnapshot: State snapshots

This module implements the "Digital Twin Engine" architecture from digitaltwin.md
"""

from .twin_engine import (
    DigitalTwinEngine,
    TwinState,
    TwinAlert,
    TwinSnapshot,
    UpdateType,
    create_digital_twin,
    demo_digital_twin
)

__all__ = [
    "DigitalTwinEngine",
    "TwinState",
    "TwinAlert",
    "TwinSnapshot",
    "UpdateType",
    "create_digital_twin",
    "demo_digital_twin"
]
