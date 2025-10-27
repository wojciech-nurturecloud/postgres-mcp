"""Monolith PostgreSQL tools for querying operational database tables.

These tools query the NCT Monolith PostgreSQL database (separate from DAPI).
The monolith database contains operational tables for users, agents, properties,
relationships, and entity transfer tracking.

Key Tables:
- fts_users: Full-text search indexed user table
- agent_contact_relationship: Many-to-many agent-contact relationships
- agent_transfer_entity_ledger: Entity ownership transfer audit trail
- fts_properties: Full-text search indexed property table
"""

from . import monolith_tools

__all__ = ["monolith_tools"]
