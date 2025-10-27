"""Monolith PostgreSQL tools for querying operational database tables.

These tools query the NCT Monolith PostgreSQL database's operational tables.
Inspired by NCT Jessica MCP's monolith profile tools.

Key Tables Queried:
- fts_users: Users with full-text search capabilities
- agent_contact_relationship: Bidirectional agent-contact relationships
- agent_transfer_entity_ledger: Entity ownership transfer audit trail
"""

import json
import logging
from typing import Any

import mcp.types as types
from pydantic import Field

from ..sql import SafeSqlDriver, SqlDriver

logger = logging.getLogger(__name__)

ResponseType = list[types.TextContent | types.ImageContent | types.EmbeddedResource]


def format_text_response(text: Any) -> ResponseType:
    """Format a text response."""
    return [types.TextContent(type="text", text=str(text))]


def format_error_response(error: str) -> ResponseType:
    """Format an error response."""
    return format_text_response(f"Error: {error}")


# ============================================================================
# USER TOOLS - Query fts_users table
# ============================================================================


async def fetch_user_by_id(
    user_id: str = Field(description="User UUID"),
    sql_driver: SqlDriver | None = None,
) -> ResponseType:
    """
    Fetch a single user by ID from the fts_users table.

    The fts_users table contains user records with full-text search indexing
    for efficient name and email searches.

    Returns:
        Complete user record including id, email, first_name, last_name, business_fk, etc.
    """
    try:
        if not sql_driver:
            raise ValueError("sql_driver is required")

        query = """
            SELECT
                id,
                email,
                first_name,
                last_name,
                business_fk,
                archived,
                is_internal,
                phone_number,
                created,
                modified,
                search_order
            FROM fts_users
            WHERE id = {}::uuid
        """

        rows = await SafeSqlDriver.execute_param_query(sql_driver, query, [user_id])

        if not rows or len(rows) == 0:
            return format_error_response(f"User not found with id: {user_id}")

        user = dict(rows[0].cells)
        return format_text_response({"user": user})

    except Exception as e:
        logger.error(f"Error fetching user by id {user_id}: {e}")
        return format_error_response(str(e))


async def fetch_user_by_email(
    email: str = Field(description="User email address"),
    sql_driver: SqlDriver | None = None,
) -> ResponseType:
    """
    Fetch a user by their email address from the fts_users table.

    Email lookup is useful for authentication, user identification, and support queries.

    Returns:
        User record if found, error if not found or multiple users with same email
    """
    try:
        if not sql_driver:
            raise ValueError("sql_driver is required")

        query = """
            SELECT
                id,
                email,
                first_name,
                last_name,
                business_fk,
                archived,
                is_internal,
                phone_number,
                created,
                modified,
                search_order
            FROM fts_users
            WHERE LOWER(email) = LOWER({})
            LIMIT 2
        """

        rows = await SafeSqlDriver.execute_param_query(sql_driver, query, [email])

        if not rows or len(rows) == 0:
            return format_error_response(f"User not found with email: {email}")

        if len(rows) > 1:
            return format_error_response(f"Multiple users found with email: {email}. Use fetch_users to see all.")

        user = dict(rows[0].cells)
        return format_text_response({"user": user})

    except Exception as e:
        logger.error(f"Error fetching user by email {email}: {e}")
        return format_error_response(str(e))


async def fetch_users(
    business_fk: str | None = Field(description="Optional business UUID filter", default=None),
    text_search: str | None = Field(description="Optional text search (name or email)", default=None),
    archived: bool | None = Field(description="Optional filter by archived status", default=None),
    is_internal: bool | None = Field(description="Optional filter for internal users (staff/agents)", default=None),
    limit: int = Field(description="Maximum results to return (1-1000)", default=100),
    offset: int = Field(description="Pagination offset", default=0),
    sql_driver: SqlDriver | None = None,
) -> ResponseType:
    """
    Search and fetch users from fts_users table with filters and pagination.

    The fts_users table supports full-text search on names and emails.
    Use this for user directory, search, and listing operations.

    Filters:
        - business_fk: Filter by business (tenant)
        - text_search: Search in first_name, last_name, email (uses ILIKE)
        - archived: Include/exclude archived users
        - is_internal: Filter for staff/agents vs. contacts

    Returns:
        Paginated list of users matching the filters
    """
    try:
        if not sql_driver:
            raise ValueError("sql_driver is required")

        # Validate limit
        if limit < 1 or limit > 1000:
            return format_error_response("Limit must be between 1 and 1000")

        # Build query
        where_clauses = []
        params = []

        if business_fk:
            where_clauses.append("business_fk = {}::uuid")
            params.append(business_fk)

        if text_search:
            where_clauses.append("(first_name ILIKE {} OR last_name ILIKE {} OR email ILIKE {})")
            search_pattern = f"%{text_search}%"
            params.extend([search_pattern, search_pattern, search_pattern])

        if archived is not None:
            where_clauses.append("archived = {}")
            params.append(archived)

        if is_internal is not None:
            where_clauses.append("is_internal = {}")
            params.append(is_internal)

        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"

        query = f"""
            SELECT
                id,
                email,
                first_name,
                last_name,
                business_fk,
                archived,
                is_internal,
                phone_number,
                created,
                modified,
                search_order
            FROM fts_users
            WHERE {where_clause}
            ORDER BY search_order ASC, last_name ASC, first_name ASC
            LIMIT {{}}
            OFFSET {{}}
        """

        params.extend([limit, offset])

        rows = await SafeSqlDriver.execute_param_query(sql_driver, query, params)

        users = [dict(row.cells) for row in rows]
        has_more = len(users) == limit

        result = {
            "users": users,
            "pagination": {"limit": limit, "offset": offset, "next_offset": offset + len(users), "has_more": has_more},
            "count": len(users),
        }

        return format_text_response(result)

    except Exception as e:
        logger.error(f"Error fetching users: {e}")
        return format_error_response(str(e))


# ============================================================================
# ACR (Agent-Contact Relationship) TOOLS - Query agent_contact_relationship table
# ============================================================================


async def fetch_acr_by_agent_id(
    agent_id: str = Field(description="Agent UUID"),
    limit: int = Field(description="Maximum results to return (1-1000)", default=100),
    offset: int = Field(description="Pagination offset", default=0),
    sql_driver: SqlDriver | None = None,
) -> ResponseType:
    """
    Fetch all agent-contact relationships for a specific agent.

    This query traverses the many-to-many relationship from agent to contacts.
    Use this to find all contacts associated with an agent.

    The agent_contact_relationship table stores:
    - grant_id: Unique relationship identifier
    - agent_id: The agent UUID
    - contact_id: The contact (user) UUID
    - business_id: Business context
    - last_touched: Last relationship modification

    Returns:
        List of relationships showing which contacts are associated with this agent
    """
    try:
        if not sql_driver:
            raise ValueError("sql_driver is required")

        # Validate limit
        if limit < 1 or limit > 1000:
            return format_error_response("Limit must be between 1 and 1000")

        query = """
            SELECT
                grant_id,
                agent_id,
                contact_id,
                business_id,
                last_touched,
                created
            FROM agent_contact_relationship
            WHERE agent_id = {}::uuid
            ORDER BY last_touched DESC
            LIMIT {}
            OFFSET {}
        """

        rows = await SafeSqlDriver.execute_param_query(sql_driver, query, [agent_id, limit, offset])

        relationships = [dict(row.cells) for row in rows]
        has_more = len(relationships) == limit

        result = {
            "relationships": relationships,
            "agent_id": agent_id,
            "pagination": {"limit": limit, "offset": offset, "next_offset": offset + len(relationships), "has_more": has_more},
            "count": len(relationships),
        }

        return format_text_response(result)

    except Exception as e:
        logger.error(f"Error fetching ACR by agent_id {agent_id}: {e}")
        return format_error_response(str(e))


async def fetch_acr_by_contact_id(
    contact_id: str = Field(description="Contact UUID"),
    limit: int = Field(description="Maximum results to return (1-1000)", default=100),
    offset: int = Field(description="Pagination offset", default=0),
    sql_driver: SqlDriver | None = None,
) -> ResponseType:
    """
    Fetch all agent-contact relationships for a specific contact.

    This query traverses the many-to-many relationship from contact to agents.
    Use this to find all agents associated with a contact.

    Bidirectional queries:
    - fetch_acr_by_agent_id: Agent → Contacts
    - fetch_acr_by_contact_id: Contact → Agents (this tool)

    Returns:
        List of relationships showing which agents are associated with this contact
    """
    try:
        if not sql_driver:
            raise ValueError("sql_driver is required")

        # Validate limit
        if limit < 1 or limit > 1000:
            return format_error_response("Limit must be between 1 and 1000")

        query = """
            SELECT
                grant_id,
                agent_id,
                contact_id,
                business_id,
                last_touched,
                created
            FROM agent_contact_relationship
            WHERE contact_id = {}::uuid
            ORDER BY last_touched DESC
            LIMIT {}
            OFFSET {}
        """

        rows = await SafeSqlDriver.execute_param_query(sql_driver, query, [contact_id, limit, offset])

        relationships = [dict(row.cells) for row in rows]
        has_more = len(relationships) == limit

        result = {
            "relationships": relationships,
            "contact_id": contact_id,
            "pagination": {"limit": limit, "offset": offset, "next_offset": offset + len(relationships), "has_more": has_more},
            "count": len(relationships),
        }

        return format_text_response(result)

    except Exception as e:
        logger.error(f"Error fetching ACR by contact_id {contact_id}: {e}")
        return format_error_response(str(e))


# ============================================================================
# ATEL (Agent Transfer Entity Ledger) TOOLS - Query agent_transfer_entity_ledger table
# ============================================================================


async def fetch_atel_by_from_entity_id_and_agent_id(
    from_entity_id: str = Field(description="Source entity UUID (the entity being transferred FROM)"),
    agent_id: str = Field(description="Agent UUID involved in the transfer"),
    limit: int = Field(description="Maximum results to return (1-1000)", default=100),
    offset: int = Field(description="Pagination offset", default=0),
    sql_driver: SqlDriver | None = None,
) -> ResponseType:
    """
    Fetch entity transfers from a specific entity by a specific agent.

    The agent_transfer_entity_ledger tracks ownership transfers with:
    - from_entity_id: Original owner/source
    - to_entity_id: New owner/destination
    - agent_id: Agent who initiated/processed the transfer
    - entity_type: What was transferred (Property, Contact, etc.)
    - created: When the transfer was recorded
    - applied: When the transfer took effect

    Use this for: "What did Agent X transfer from Entity Y?"

    Returns:
        List of transfer records matching both from_entity_id and agent_id
    """
    try:
        if not sql_driver:
            raise ValueError("sql_driver is required")

        # Validate limit
        if limit < 1 or limit > 1000:
            return format_error_response("Limit must be between 1 and 1000")

        query = """
            SELECT
                id,
                from_entity_id,
                to_entity_id,
                agent_id,
                entity_type,
                created,
                applied,
                metadata
            FROM agent_transfer_entity_ledger
            WHERE from_entity_id = {}::uuid
              AND agent_id = {}::uuid
            ORDER BY created DESC, id
            LIMIT {}
            OFFSET {}
        """

        rows = await SafeSqlDriver.execute_param_query(sql_driver, query, [from_entity_id, agent_id, limit, offset])

        records = [dict(row.cells) for row in rows]
        has_more = len(records) == limit

        result = {
            "records": records,
            "from_entity_id": from_entity_id,
            "agent_id": agent_id,
            "pagination": {"limit": limit, "offset": offset, "next_offset": offset + len(records), "has_more": has_more},
            "count": len(records),
        }

        return format_text_response(result)

    except Exception as e:
        logger.error(f"Error fetching ATEL by from_entity_id={from_entity_id} and agent_id={agent_id}: {e}")
        return format_error_response(str(e))


async def fetch_atel_by_from_entity_id(
    from_entity_id: str = Field(description="Source entity UUID (the entity being transferred FROM)"),
    limit: int = Field(description="Maximum results to return (1-1000)", default=100),
    offset: int = Field(description="Pagination offset", default=0),
    sql_driver: SqlDriver | None = None,
) -> ResponseType:
    """
    Fetch all transfers FROM a specific entity.

    Use this for: "Show me the complete transfer history of Entity X"

    This queries the audit trail to see:
    - Who received the entity (to_entity_id)
    - Which agent processed each transfer
    - When transfers occurred
    - What entity type was transferred

    Returns:
        Complete transfer history for entities that originated from from_entity_id
    """
    try:
        if not sql_driver:
            raise ValueError("sql_driver is required")

        # Validate limit
        if limit < 1 or limit > 1000:
            return format_error_response("Limit must be between 1 and 1000")

        query = """
            SELECT
                id,
                from_entity_id,
                to_entity_id,
                agent_id,
                entity_type,
                created,
                applied,
                metadata
            FROM agent_transfer_entity_ledger
            WHERE from_entity_id = {}::uuid
            ORDER BY created DESC, id
            LIMIT {}
            OFFSET {}
        """

        rows = await SafeSqlDriver.execute_param_query(sql_driver, query, [from_entity_id, limit, offset])

        records = [dict(row.cells) for row in rows]
        has_more = len(records) == limit

        result = {
            "records": records,
            "from_entity_id": from_entity_id,
            "pagination": {"limit": limit, "offset": offset, "next_offset": offset + len(records), "has_more": has_more},
            "count": len(records),
        }

        return format_text_response(result)

    except Exception as e:
        logger.error(f"Error fetching ATEL by from_entity_id={from_entity_id}: {e}")
        return format_error_response(str(e))


async def fetch_atel_by_to_entity_id(
    to_entity_id: str = Field(description="Destination entity UUID (the entity receiving transfers)"),
    limit: int = Field(description="Maximum results to return (1-1000)", default=100),
    offset: int = Field(description="Pagination offset", default=0),
    sql_driver: SqlDriver | None = None,
) -> ResponseType:
    """
    Fetch all transfers TO a specific entity.

    Use this for: "What entities were transferred INTO Agent/Entity X's portfolio?"

    This queries the reverse direction from fetch_atel_by_from_entity_id:
    - What was received (from_entity_id)
    - Which agent processed each transfer
    - When transfers occurred

    Bidirectional audit trail:
    - fetch_atel_by_from_entity_id: What went OUT
    - fetch_atel_by_to_entity_id: What came IN (this tool)

    Returns:
        List of transfers where to_entity_id is the destination
    """
    try:
        if not sql_driver:
            raise ValueError("sql_driver is required")

        # Validate limit
        if limit < 1 or limit > 1000:
            return format_error_response("Limit must be between 1 and 1000")

        query = """
            SELECT
                id,
                from_entity_id,
                to_entity_id,
                agent_id,
                entity_type,
                created,
                applied,
                metadata
            FROM agent_transfer_entity_ledger
            WHERE to_entity_id = {}::uuid
            ORDER BY created DESC, id
            LIMIT {}
            OFFSET {}
        """

        rows = await SafeSqlDriver.execute_param_query(sql_driver, query, [to_entity_id, limit, offset])

        records = [dict(row.cells) for row in rows]
        has_more = len(records) == limit

        result = {
            "records": records,
            "to_entity_id": to_entity_id,
            "pagination": {"limit": limit, "offset": offset, "next_offset": offset + len(records), "has_more": has_more},
            "count": len(records),
        }

        return format_text_response(result)

    except Exception as e:
        logger.error(f"Error fetching ATEL by to_entity_id={to_entity_id}: {e}")
        return format_error_response(str(e))


# ============================================================================
# PROPERTY TOOLS - Query fts_properties table
# ============================================================================


async def fetch_property_by_id(
    property_id: str = Field(description="Property UUID"),
    sql_driver: SqlDriver | None = None,
) -> ResponseType:
    """
    Fetch a single property by ID from the fts_properties table.

    The fts_properties table contains property records with full-text search indexing
    for efficient address and agent searches. Includes vendor details, phase, agents,
    and lead scoring information.

    Returns:
        Complete property record including address, agents, vendor info, phase, lead score, etc.
    """
    try:
        if not sql_driver:
            raise ValueError("sql_driver is required")

        query = """
            SELECT
                property_id,
                org_name,
                sales_manager_id,
                agent_id,
                secondary_agent_id,
                sales_assistant_id,
                support_agent_id,
                vendor_id,
                phase,
                suburb_string,
                address,
                agent_full_name,
                vendor_full_name,
                vendor_phone,
                vendor_email,
                lead_visible,
                lead_score,
                last_contacted,
                business_fk,
                team_fk,
                agency_fk,
                team_name,
                image_url,
                completed_appraisal_dt,
                vendor_medium,
                is_db,
                lead_score_order,
                vendor_first_name,
                vendor_last_name,
                is_database_state,
                archived_status
            FROM fts_properties
            WHERE property_id = {}::uuid
        """

        rows = await SafeSqlDriver.execute_param_query(sql_driver, query, [property_id])

        if not rows or len(rows) == 0:
            return format_error_response(f"Property not found with id: {property_id}")

        property_data = dict(rows[0].cells)
        return format_text_response({"property": property_data})

    except Exception as e:
        logger.error(f"Error fetching property by id {property_id}: {e}")
        return format_error_response(str(e))


async def fetch_properties(
    business_fk: str | None = Field(description="Optional business UUID filter", default=None),
    address: str | None = Field(description="Optional text search for address", default=None),
    limit: int = Field(description="Maximum results to return (1-1000)", default=100),
    offset: int = Field(description="Pagination offset", default=0),
    sql_driver: SqlDriver | None = None,
) -> ResponseType:
    """
    Search and fetch properties from fts_properties table with filters and pagination.

    The fts_properties table supports full-text search on addresses and contains
    comprehensive property data including agents, vendors, phase, and lead scoring.

    Filters:
        - business_fk: Filter by business (tenant)
        - address: Search in address field (uses ILIKE)

    Sorting:
        Results ordered by lead_score_order (priority) then address

    Returns:
        Paginated list of properties matching the filters
    """
    try:
        if not sql_driver:
            raise ValueError("sql_driver is required")

        # Validate limit
        if limit < 1 or limit > 1000:
            return format_error_response("Limit must be between 1 and 1000")

        # Build query
        where_clauses = []
        params = []

        if business_fk:
            where_clauses.append("business_fk = {}::uuid")
            params.append(business_fk)

        if address:
            where_clauses.append("address ILIKE {}")
            params.append(f"%{address}%")

        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"

        query = f"""
            SELECT
                property_id,
                org_name,
                sales_manager_id,
                agent_id,
                secondary_agent_id,
                sales_assistant_id,
                support_agent_id,
                vendor_id,
                phase,
                suburb_string,
                address,
                agent_full_name,
                vendor_full_name,
                vendor_phone,
                vendor_email,
                lead_visible,
                lead_score,
                last_contacted,
                business_fk,
                team_fk,
                agency_fk,
                team_name,
                image_url,
                completed_appraisal_dt,
                vendor_medium,
                is_db,
                lead_score_order,
                vendor_first_name,
                vendor_last_name,
                is_database_state,
                archived_status
            FROM fts_properties
            WHERE {where_clause}
            ORDER BY lead_score_order, address
            LIMIT {{}}
            OFFSET {{}}
        """

        params.extend([limit, offset])

        rows = await SafeSqlDriver.execute_param_query(sql_driver, query, params)

        properties = [dict(row.cells) for row in rows]
        has_more = len(properties) == limit

        result = {
            "properties": properties,
            "pagination": {"limit": limit, "offset": offset, "next_offset": offset + len(properties), "has_more": has_more},
            "count": len(properties),
        }

        return format_text_response(result)

    except Exception as e:
        logger.error(f"Error fetching properties: {e}")
        return format_error_response(str(e))


# ============================================================================
# AGENT TOOLS - Query fts_users table (where is_internal = true)
# ============================================================================


async def fetch_agent_by_id(
    agent_id: str = Field(description="Agent UUID"),
    sql_driver: SqlDriver | None = None,
) -> ResponseType:
    """
    Fetch a single agent by ID from the fts_users table.

    Agents are users with is_internal = true. This distinguishes staff/agents
    from contacts/customers in the system.

    Note: Uses user_id column (not id) for agent lookups.

    Returns:
        Agent record including email, full_name, phone, business, active status
    """
    try:
        if not sql_driver:
            raise ValueError("sql_driver is required")

        query = """
            SELECT
                user_id,
                org_name,
                full_name,
                phone_number,
                email,
                business_fk,
                is_agent_active,
                archived
            FROM fts_users
            WHERE user_id = {}::uuid
              AND is_internal = true
        """

        rows = await SafeSqlDriver.execute_param_query(sql_driver, query, [agent_id])

        if not rows or len(rows) == 0:
            return format_error_response(f"Agent not found with id: {agent_id}")

        agent = dict(rows[0].cells)
        return format_text_response({"agent": agent})

    except Exception as e:
        logger.error(f"Error fetching agent by id {agent_id}: {e}")
        return format_error_response(str(e))


async def fetch_agent_by_email(
    email: str = Field(description="Agent email address"),
    sql_driver: SqlDriver | None = None,
) -> ResponseType:
    """
    Fetch an agent by their email address from the fts_users table.

    Only returns users where is_internal = true (staff/agents).
    Useful for agent authentication, lookup, and support queries.

    Returns:
        Agent record if found, error if not found or email belongs to non-agent
    """
    try:
        if not sql_driver:
            raise ValueError("sql_driver is required")

        query = """
            SELECT
                user_id,
                org_name,
                full_name,
                phone_number,
                email,
                business_fk,
                is_agent_active,
                archived
            FROM fts_users
            WHERE LOWER(email) = LOWER({})
              AND is_internal = true
            LIMIT 2
        """

        rows = await SafeSqlDriver.execute_param_query(sql_driver, query, [email])

        if not rows or len(rows) == 0:
            return format_error_response(f"Agent not found with email: {email}")

        if len(rows) > 1:
            return format_error_response(f"Multiple agents found with email: {email}. Use fetch_agents to see all.")

        agent = dict(rows[0].cells)
        return format_text_response({"agent": agent})

    except Exception as e:
        logger.error(f"Error fetching agent by email {email}: {e}")
        return format_error_response(str(e))


async def fetch_agents(
    business_id: str | None = Field(description="Optional business UUID filter", default=None),
    limit: int = Field(description="Maximum results to return (1-1000)", default=100),
    offset: int = Field(description="Pagination offset", default=0),
    sql_driver: SqlDriver | None = None,
) -> ResponseType:
    """
    Fetch agents from fts_users table with filters and pagination.

    Agents are users where is_internal = true. This tool provides agent directory
    and listing functionality with business filtering.

    Filters:
        - business_id: Filter by business (tenant)

    Sorting:
        Results ordered by org_name then full_name

    Returns:
        Paginated list of agents matching the filters
    """
    try:
        if not sql_driver:
            raise ValueError("sql_driver is required")

        # Validate limit
        if limit < 1 or limit > 1000:
            return format_error_response("Limit must be between 1 and 1000")

        # Build query - always filter for agents
        where_clauses = ["is_internal = true"]
        params = []

        if business_id:
            where_clauses.append("business_fk = {}::uuid")
            params.append(business_id)

        where_clause = " AND ".join(where_clauses)

        query = f"""
            SELECT
                user_id,
                org_name,
                full_name,
                phone_number,
                email,
                business_fk,
                is_agent_active,
                archived
            FROM fts_users
            WHERE {where_clause}
            ORDER BY org_name, full_name
            LIMIT {{}}
            OFFSET {{}}
        """

        params.extend([limit, offset])

        rows = await SafeSqlDriver.execute_param_query(sql_driver, query, params)

        agents = [dict(row.cells) for row in rows]
        has_more = len(agents) == limit

        result = {
            "agents": agents,
            "pagination": {"limit": limit, "offset": offset, "next_offset": offset + len(agents), "has_more": has_more},
            "count": len(agents),
        }

        return format_text_response(result)

    except Exception as e:
        logger.error(f"Error fetching agents: {e}")
        return format_error_response(str(e))
