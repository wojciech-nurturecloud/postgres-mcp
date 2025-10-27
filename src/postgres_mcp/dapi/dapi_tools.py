"""DAPI (Data API) tools for querying historical data from data_api_v2.historical_data table.

DAPI is NurtureCloud's Data API - a message bus system that stores JSON payloads in PostgreSQL.
The data is stored in the data_api_v2.historical_data table, partitioned by component_name.

Entity Types (component_name values):
- contact: Contacts/people in the system
- listing: Property listings (sales lifecycle)
- agent: Staff users/agents
- appointment: Property viewings (appraisals, inspections, auctions)
- interaction: Communications (phone, email, SMS)
- offer: Offers made for listings
- interestedbuyer: Buyers interested in properties
- agenttocontactrelationship: Agent-contact relationships
- business: Top-level organizational units
- agency: Sales entities within organizations
- And 20+ more entity types

Table Structure:
- id: Entity UUID
- org_name: Organization name (tenant identifier)
- business_id: Business UUID (sub-tenant)
- agency_id: Agency UUID
- component_name: Entity type (Contact, Listing, etc.)
- payload: Full entity JSON blob
- correlation_id: Source tracking
- is_deleted: Soft delete flag
- delete_is_merge: True if entity was merged
- merge_payload: Target entity when merged
- last_touched: Last modification timestamp
- row_number: Pagination cursor
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


async def query_historical_data(
    component_name: str = Field(description="Entity type (e.g., 'contact', 'listing', 'agent', 'offer')"),
    org_name: str = Field(description="Organization name"),
    business_id: str | None = Field(description="Optional business UUID filter", default=None),
    agency_id: str | None = Field(description="Optional agency UUID filter", default=None),
    offset: int = Field(description="Pagination offset (row_number to start from)", default=0),
    limit: int = Field(description="Maximum number of records to return (1-1000)", default=100),
    since: str | None = Field(description="Optional ISO timestamp - only return entities modified after this time", default=None),
    include_deleted: bool = Field(description="Include soft-deleted entities", default=False),
    sql_driver: SqlDriver | None = None,
) -> ResponseType:
    """
    Query DAPI historical data with pagination and filtering.

    This is the main workhorse tool for fetching entities from DAPI's historical_data table.
    Returns paginated results with a cursor (max row_number) for fetching the next page.

    Examples:
        - Fetch contacts: component_name='contact', org_name='MyOrg'
        - Fetch listings for a business: component_name='listing', org_name='MyOrg', business_id='uuid'
        - Fetch recent changes: component_name='contact', org_name='MyOrg', since='2025-01-01T00:00:00Z'

    Returns:
        JSON with:
        - entities: List of entity records (id, payload, last_touched, etc.)
        - pagination: {offset, limit, next_offset, has_more}
        - count: Number of entities returned
    """
    try:
        if not sql_driver:
            raise ValueError("sql_driver is required")

        # Validate limit
        if limit < 1 or limit > 1000:
            return format_error_response("Limit must be between 1 and 1000")

        # Build query
        where_clauses = [
            "component_name = {}",
            "org_name = {}",
        ]
        params = [component_name, org_name]

        if business_id:
            where_clauses.append("business_id = {}::uuid")
            params.append(business_id)

        if agency_id:
            where_clauses.append("agency_id = {}::uuid")
            params.append(agency_id)

        if offset > 0:
            where_clauses.append("row_number > {}")
            params.append(offset)

        if since:
            where_clauses.append("last_touched >= {}::timestamp")
            params.append(since)

        if not include_deleted:
            where_clauses.append("(is_deleted = false OR is_deleted IS NULL)")

        where_clause = " AND ".join(where_clauses)

        query = f"""
            SELECT
                id,
                org_name,
                business_id,
                agency_id,
                component_name,
                payload,
                source,
                correlation_id,
                is_deleted,
                delete_is_merge,
                merge_payload,
                last_touched,
                row_number
            FROM data_api_v2.historical_data
            WHERE {where_clause}
            ORDER BY row_number ASC
            LIMIT {{}}
        """

        params.append(limit)

        rows = await SafeSqlDriver.execute_param_query(sql_driver, query, params)

        if not rows:
            return format_text_response(
                {"entities": [], "pagination": {"offset": offset, "limit": limit, "next_offset": offset, "has_more": False}, "count": 0}
            )

        # Convert rows to dictionaries
        entities = []
        max_row_number = offset

        for row in rows:
            entity = dict(row.cells)
            # Parse payload JSON if it's a string
            if isinstance(entity.get("payload"), str):
                try:
                    entity["payload"] = json.loads(entity["payload"])
                except json.JSONDecodeError:
                    pass  # Leave as string if not valid JSON

            # Parse merge_payload JSON if it's a string
            if isinstance(entity.get("merge_payload"), str):
                try:
                    entity["merge_payload"] = json.loads(entity["merge_payload"])
                except json.JSONDecodeError:
                    pass

            # Track max row_number for pagination
            if entity.get("row_number") and entity["row_number"] > max_row_number:
                max_row_number = entity["row_number"]

            entities.append(entity)

        # Determine if there are more results
        has_more = len(entities) == limit

        result = {"entities": entities, "pagination": {"offset": offset, "limit": limit, "next_offset": max_row_number, "has_more": has_more}, "count": len(entities)}

        return format_text_response(result)

    except Exception as e:
        logger.error(f"Error querying historical data: {e}")
        return format_error_response(str(e))


async def fetch_entity_by_id(
    component_name: str = Field(description="Entity type (e.g., 'contact', 'listing', 'agent')"),
    org_name: str = Field(description="Organization name"),
    entity_id: str = Field(description="Entity UUID"),
    sql_driver: SqlDriver | None = None,
) -> ResponseType:
    """
    Fetch a single entity by its ID from DAPI historical data.

    Fast lookup for a specific entity when you know its ID.

    Examples:
        - Fetch contact: component_name='contact', org_name='MyOrg', entity_id='uuid'
        - Fetch listing: component_name='listing', org_name='MyOrg', entity_id='uuid'

    Returns:
        JSON with the full entity record including payload
    """
    try:
        if not sql_driver:
            raise ValueError("sql_driver is required")

        query = """
            SELECT
                id,
                org_name,
                business_id,
                agency_id,
                component_name,
                payload,
                source,
                correlation_id,
                is_deleted,
                delete_is_merge,
                merge_payload,
                last_touched,
                row_number
            FROM data_api_v2.historical_data
            WHERE component_name = {}
              AND org_name = {}
              AND id = {}::uuid
        """

        rows = await SafeSqlDriver.execute_param_query(sql_driver, query, [component_name, org_name, entity_id])

        if not rows or len(rows) == 0:
            return format_error_response(f"Entity not found: {component_name} with id {entity_id}")

        entity = dict(rows[0].cells)

        # Parse payload JSON if it's a string
        if isinstance(entity.get("payload"), str):
            try:
                entity["payload"] = json.loads(entity["payload"])
            except json.JSONDecodeError:
                pass

        # Parse merge_payload JSON if it's a string
        if isinstance(entity.get("merge_payload"), str):
            try:
                entity["merge_payload"] = json.loads(entity["merge_payload"])
            except json.JSONDecodeError:
                pass

        return format_text_response(entity)

    except Exception as e:
        logger.error(f"Error fetching entity by id: {e}")
        return format_error_response(str(e))


async def extract_payload_field(
    component_name: str = Field(description="Entity type (e.g., 'contact', 'listing')"),
    org_name: str = Field(description="Organization name"),
    entity_id: str = Field(description="Entity UUID"),
    field_path: str = Field(description="Dot-notation path to field in payload (e.g., 'data.firstName' or 'organisationName')"),
    sql_driver: SqlDriver | None = None,
) -> ResponseType:
    """
    Extract a specific field from an entity's JSON payload using dot notation.

    Useful for navigating deeply nested JSON payloads without fetching the entire entity.

    Examples:
        - Get contact's first name: field_path='data.firstName' (if payload has nested data)
        - Get contact's email: field_path='email'
        - Get listing price: field_path='price.amount'

    Note: The exact field paths depend on the entity type and DAPI schema version.

    Returns:
        The value at the specified field path
    """
    try:
        if not sql_driver:
            raise ValueError("sql_driver is required")

        # Convert dot notation to PostgreSQL JSON path
        # e.g., "data.firstName" -> "payload->'data'->>'firstName'"
        path_parts = field_path.split(".")

        if len(path_parts) == 0:
            return format_error_response("field_path cannot be empty")

        # Build JSON accessor
        # For all but the last part, use -> (returns JSON)
        # For the last part, use ->> (returns text)
        json_path = "payload"
        for i, part in enumerate(path_parts):
            if i == len(path_parts) - 1:
                json_path += f"->'{part}'"
            else:
                json_path += f"->'{part}'"

        query = f"""
            SELECT
                id,
                {json_path} as field_value
            FROM data_api_v2.historical_data
            WHERE component_name = {{}}
              AND org_name = {{}}
              AND id = {{}}::uuid
        """

        rows = await SafeSqlDriver.execute_param_query(sql_driver, query, [component_name, org_name, entity_id])

        if not rows or len(rows) == 0:
            return format_error_response(f"Entity not found: {component_name} with id {entity_id}")

        field_value = rows[0].cells.get("field_value")

        # Try to parse as JSON if it looks like JSON
        if isinstance(field_value, str):
            try:
                field_value = json.loads(field_value)
            except json.JSONDecodeError:
                pass  # Return as string

        result = {"entity_id": entity_id, "component_name": component_name, "field_path": field_path, "value": field_value}

        return format_text_response(result)

    except Exception as e:
        logger.error(f"Error extracting payload field: {e}")
        return format_error_response(str(e))


async def query_by_correlation_id(
    component_name: str = Field(description="Entity type (e.g., 'contact', 'listing')"),
    org_name: str = Field(description="Organization name"),
    source: str = Field(description="External source system (e.g., 'REA', 'Domain')"),
    correlation_id: str = Field(description="External correlation/source ID"),
    sql_driver: SqlDriver | None = None,
) -> ResponseType:
    """
    Find entities by their external correlation ID.

    DAPI tracks the source of each entity via correlation_id. This tool finds entities
    that originated from external systems (e.g., REA, Domain) by their external ID.

    Examples:
        - Find REA listing: component_name='listing', source='REA', correlation_id='external-id-123'
        - Find external contact: component_name='contact', source='Domain', correlation_id='contact-456'

    Returns:
        List of entities matching the correlation criteria
    """
    try:
        if not sql_driver:
            raise ValueError("sql_driver is required")

        query = """
            SELECT
                id,
                org_name,
                business_id,
                agency_id,
                component_name,
                payload,
                source,
                correlation_id,
                is_deleted,
                delete_is_merge,
                merge_payload,
                last_touched,
                row_number
            FROM data_api_v2.historical_data
            WHERE component_name = {}
              AND org_name = {}
              AND source = {}
              AND correlation_id = {}
            ORDER BY last_touched DESC
        """

        rows = await SafeSqlDriver.execute_param_query(sql_driver, query, [component_name, org_name, source, correlation_id])

        if not rows:
            return format_text_response({"entities": [], "count": 0, "message": f"No entities found with source={source}, correlation_id={correlation_id}"})

        entities = []
        for row in rows:
            entity = dict(row.cells)

            # Parse payload JSON if it's a string
            if isinstance(entity.get("payload"), str):
                try:
                    entity["payload"] = json.loads(entity["payload"])
                except json.JSONDecodeError:
                    pass

            # Parse merge_payload JSON if it's a string
            if isinstance(entity.get("merge_payload"), str):
                try:
                    entity["merge_payload"] = json.loads(entity["merge_payload"])
                except json.JSONDecodeError:
                    pass

            entities.append(entity)

        result = {"entities": entities, "count": len(entities), "search": {"component_name": component_name, "source": source, "correlation_id": correlation_id}}

        return format_text_response(result)

    except Exception as e:
        logger.error(f"Error querying by correlation_id: {e}")
        return format_error_response(str(e))


async def list_component_names(
    org_name: str = Field(description="Organization name"),
    sql_driver: SqlDriver | None = None,
) -> ResponseType:
    """
    List all available component types (entity types) in DAPI for an organization.

    Discovery tool to see what types of entities exist in the system.
    Common types: contact, listing, agent, offer, appointment, interaction, interestedbuyer, etc.

    Returns:
        List of available component_name values with counts
    """
    try:
        if not sql_driver:
            raise ValueError("sql_driver is required")

        query = """
            SELECT
                component_name,
                COUNT(*) as total_count,
                COUNT(*) FILTER (WHERE is_deleted = true) as deleted_count,
                MAX(last_touched) as last_modified
            FROM data_api_v2.historical_data
            WHERE org_name = {}
            GROUP BY component_name
            ORDER BY total_count DESC, component_name ASC
        """

        rows = await SafeSqlDriver.execute_param_query(sql_driver, query, [org_name])

        if not rows:
            return format_text_response({"components": [], "count": 0, "message": f"No components found for org_name={org_name}"})

        components = []
        for row in rows:
            components.append(
                {
                    "component_name": row.cells["component_name"],
                    "total_count": row.cells["total_count"],
                    "deleted_count": row.cells["deleted_count"],
                    "active_count": row.cells["total_count"] - row.cells["deleted_count"],
                    "last_modified": str(row.cells["last_modified"]) if row.cells.get("last_modified") else None,
                }
            )

        result = {"org_name": org_name, "components": components, "total_component_types": len(components)}

        return format_text_response(result)

    except Exception as e:
        logger.error(f"Error listing component names: {e}")
        return format_error_response(str(e))


async def query_deleted_entities(
    component_name: str = Field(description="Entity type (e.g., 'contact', 'listing')"),
    org_name: str = Field(description="Organization name"),
    business_id: str | None = Field(description="Optional business UUID filter", default=None),
    since: str | None = Field(description="Optional ISO timestamp - only return entities deleted after this time", default=None),
    include_merge_targets: bool = Field(description="Include merge_payload for merged entities", default=True),
    limit: int = Field(description="Maximum number of records to return (1-1000)", default=100),
    sql_driver: SqlDriver | None = None,
) -> ResponseType:
    """
    Query deleted entities from DAPI historical data.

    Track entity deletions and merges. When include_merge_targets=true, also returns
    the target entity for merge operations (when one entity is consolidated into another).

    Examples:
        - Find deleted contacts: component_name='contact', org_name='MyOrg'
        - Find recent deletions: component_name='listing', since='2025-01-01T00:00:00Z'

    Returns:
        List of deleted entities with merge information if available
    """
    try:
        if not sql_driver:
            raise ValueError("sql_driver is required")

        # Validate limit
        if limit < 1 or limit > 1000:
            return format_error_response("Limit must be between 1 and 1000")

        where_clauses = ["component_name = {}", "org_name = {}", "is_deleted = true"]
        params = [component_name, org_name]

        if business_id:
            where_clauses.append("business_id = {}::uuid")
            params.append(business_id)

        if since:
            where_clauses.append("last_touched >= {}::timestamp")
            params.append(since)

        where_clause = " AND ".join(where_clauses)

        select_fields = """
            id,
            org_name,
            business_id,
            agency_id,
            component_name,
            payload,
            source,
            correlation_id,
            is_deleted,
            delete_is_merge,
            last_touched,
            row_number
        """

        if include_merge_targets:
            select_fields += ",\n            merge_payload"

        query = f"""
            SELECT {select_fields}
            FROM data_api_v2.historical_data
            WHERE {where_clause}
            ORDER BY last_touched DESC
            LIMIT {{}}
        """

        params.append(limit)

        rows = await SafeSqlDriver.execute_param_query(sql_driver, query, params)

        if not rows:
            return format_text_response({"deleted_entities": [], "count": 0, "message": "No deleted entities found"})

        entities = []
        for row in rows:
            entity = dict(row.cells)

            # Parse payload JSON if it's a string
            if isinstance(entity.get("payload"), str):
                try:
                    entity["payload"] = json.loads(entity["payload"])
                except json.JSONDecodeError:
                    pass

            # Parse merge_payload JSON if it's a string and included
            if include_merge_targets and isinstance(entity.get("merge_payload"), str):
                try:
                    entity["merge_payload"] = json.loads(entity["merge_payload"])
                except json.JSONDecodeError:
                    pass

            entities.append(entity)

        result = {"deleted_entities": entities, "count": len(entities), "component_name": component_name, "org_name": org_name}

        return format_text_response(result)

    except Exception as e:
        logger.error(f"Error querying deleted entities: {e}")
        return format_error_response(str(e))


async def batch_fetch_by_ids(
    component_name: str = Field(description="Entity type (e.g., 'contact', 'listing')"),
    org_name: str = Field(description="Organization name"),
    entity_ids: list[str] = Field(description="List of entity UUIDs (max 100)"),
    sql_driver: SqlDriver | None = None,
) -> ResponseType:
    """
    Fetch multiple entities by their IDs in a single query.

    Efficient bulk fetch operation when you have multiple entity IDs.
    Reduces round trips compared to fetching entities one by one.

    Examples:
        - Fetch multiple contacts: component_name='contact', entity_ids=['uuid1', 'uuid2', 'uuid3']
        - Fetch related listings: component_name='listing', entity_ids=['uuid1', 'uuid2']

    Returns:
        List of entities matching the provided IDs
    """
    try:
        if not sql_driver:
            raise ValueError("sql_driver is required")

        # Validate input
        if not entity_ids or len(entity_ids) == 0:
            return format_error_response("entity_ids cannot be empty")

        if len(entity_ids) > 100:
            return format_error_response("Maximum 100 entity_ids allowed per request")

        # Build query with IN clause
        # Use array parameter for safety
        query = """
            SELECT
                id,
                org_name,
                business_id,
                agency_id,
                component_name,
                payload,
                source,
                correlation_id,
                is_deleted,
                delete_is_merge,
                merge_payload,
                last_touched,
                row_number
            FROM data_api_v2.historical_data
            WHERE component_name = {}
              AND org_name = {}
              AND id = ANY({}::uuid[])
            ORDER BY row_number ASC
        """

        rows = await SafeSqlDriver.execute_param_query(sql_driver, query, [component_name, org_name, entity_ids])

        if not rows:
            return format_text_response({"entities": [], "count": 0, "requested_count": len(entity_ids), "message": "No entities found"})

        entities = []
        found_ids = set()

        for row in rows:
            entity = dict(row.cells)

            # Track which IDs were found
            if entity.get("id"):
                found_ids.add(str(entity["id"]))

            # Parse payload JSON if it's a string
            if isinstance(entity.get("payload"), str):
                try:
                    entity["payload"] = json.loads(entity["payload"])
                except json.JSONDecodeError:
                    pass

            # Parse merge_payload JSON if it's a string
            if isinstance(entity.get("merge_payload"), str):
                try:
                    entity["merge_payload"] = json.loads(entity["merge_payload"])
                except json.JSONDecodeError:
                    pass

            entities.append(entity)

        # Determine which IDs were not found
        requested_ids = set(entity_ids)
        missing_ids = list(requested_ids - found_ids)

        result = {
            "entities": entities,
            "count": len(entities),
            "requested_count": len(entity_ids),
            "found_count": len(found_ids),
            "missing_ids": missing_ids if missing_ids else None,
        }

        return format_text_response(result)

    except Exception as e:
        logger.error(f"Error batch fetching by ids: {e}")
        return format_error_response(str(e))


async def count_entities(
    component_name: str = Field(description="Entity type (e.g., 'contact', 'listing')"),
    org_name: str = Field(description="Organization name"),
    business_id: str | None = Field(description="Optional business UUID filter", default=None),
    include_deleted: bool = Field(description="Include deleted entities in count", default=False),
    sql_driver: SqlDriver | None = None,
) -> ResponseType:
    """
    Get count statistics for entities in DAPI.

    Useful for monitoring, pagination planning, and understanding data volume.

    Examples:
        - Count all contacts: component_name='contact', org_name='MyOrg'
        - Count active listings for a business: component_name='listing', business_id='uuid'

    Returns:
        Statistics including total count, deleted count, and active count
    """
    try:
        if not sql_driver:
            raise ValueError("sql_driver is required")

        where_clauses = ["component_name = {}", "org_name = {}"]
        params = [component_name, org_name]

        if business_id:
            where_clauses.append("business_id = {}::uuid")
            params.append(business_id)

        where_clause = " AND ".join(where_clauses)

        query = f"""
            SELECT
                COUNT(*) as total_count,
                COUNT(*) FILTER (WHERE is_deleted = true) as deleted_count,
                COUNT(*) FILTER (WHERE is_deleted = false OR is_deleted IS NULL) as active_count,
                COUNT(*) FILTER (WHERE delete_is_merge = true) as merged_count,
                MAX(last_touched) as last_modified,
                MIN(last_touched) as first_created
            FROM data_api_v2.historical_data
            WHERE {where_clause}
        """

        if not include_deleted:
            query += " AND (is_deleted = false OR is_deleted IS NULL)"

        rows = await SafeSqlDriver.execute_param_query(sql_driver, query, params)

        if not rows or len(rows) == 0:
            return format_text_response(
                {
                    "component_name": component_name,
                    "org_name": org_name,
                    "business_id": business_id,
                    "total_count": 0,
                    "deleted_count": 0,
                    "active_count": 0,
                    "merged_count": 0,
                }
            )

        stats = dict(rows[0].cells)

        result = {
            "component_name": component_name,
            "org_name": org_name,
            "business_id": business_id,
            "total_count": stats.get("total_count", 0),
            "deleted_count": stats.get("deleted_count", 0),
            "active_count": stats.get("active_count", 0),
            "merged_count": stats.get("merged_count", 0),
            "last_modified": str(stats["last_modified"]) if stats.get("last_modified") else None,
            "first_created": str(stats["first_created"]) if stats.get("first_created") else None,
        }

        return format_text_response(result)

    except Exception as e:
        logger.error(f"Error counting entities: {e}")
        return format_error_response(str(e))
