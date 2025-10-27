# ruff: noqa: B008
import argparse
import asyncio
import logging
import os
import signal
import sys
from enum import Enum
from typing import Any
from typing import List
from typing import Literal
from typing import Union

import mcp.types as types
from mcp.server.fastmcp import FastMCP
from pydantic import Field
from pydantic import validate_call

from postgres_mcp.index.dta_calc import DatabaseTuningAdvisor

from .artifacts import ErrorResult
from .artifacts import ExplainPlanArtifact
from .database_health import DatabaseHealthTool
from .database_health import HealthType
from .dapi import dapi_tools
from .monolith import monolith_tools
from .explain import ExplainPlanTool
from .index.index_opt_base import MAX_NUM_INDEX_TUNING_QUERIES
from .index.llm_opt import LLMOptimizerTool
from .index.presentation import TextPresentation
from .sql import DbConnPool
from .sql import SafeSqlDriver
from .sql import SqlDriver
from .sql import check_hypopg_installation_status
from .sql import obfuscate_password
from .top_queries import TopQueriesCalc

# Initialize FastMCP with default settings
mcp = FastMCP("postgres-mcp")

# Constants
PG_STAT_STATEMENTS = "pg_stat_statements"
HYPOPG_EXTENSION = "hypopg"

ResponseType = List[types.TextContent | types.ImageContent | types.EmbeddedResource]

logger = logging.getLogger(__name__)


class AccessMode(str, Enum):
    """SQL access modes for the server."""

    UNRESTRICTED = "unrestricted"  # Unrestricted access
    RESTRICTED = "restricted"  # Read-only with safety features


# Global variables
db_connection = DbConnPool()
current_access_mode = AccessMode.UNRESTRICTED
shutdown_in_progress = False


async def get_sql_driver() -> Union[SqlDriver, SafeSqlDriver]:
    """Get the appropriate SQL driver based on the current access mode."""
    base_driver = SqlDriver(conn=db_connection)

    if current_access_mode == AccessMode.RESTRICTED:
        logger.debug("Using SafeSqlDriver with restrictions (RESTRICTED mode)")
        return SafeSqlDriver(sql_driver=base_driver, timeout=30)  # 30 second timeout
    else:
        logger.debug("Using unrestricted SqlDriver (UNRESTRICTED mode)")
        return base_driver


def format_text_response(text: Any) -> ResponseType:
    """Format a text response."""
    return [types.TextContent(type="text", text=str(text))]


def format_error_response(error: str) -> ResponseType:
    """Format an error response."""
    return format_text_response(f"Error: {error}")


@mcp.tool(description="List all schemas in the database")
async def list_schemas() -> ResponseType:
    """List all schemas in the database."""
    try:
        sql_driver = await get_sql_driver()
        rows = await sql_driver.execute_query(
            """
            SELECT
                schema_name,
                schema_owner,
                CASE
                    WHEN schema_name LIKE 'pg_%' THEN 'System Schema'
                    WHEN schema_name = 'information_schema' THEN 'System Information Schema'
                    ELSE 'User Schema'
                END as schema_type
            FROM information_schema.schemata
            ORDER BY schema_type, schema_name
            """
        )
        schemas = [row.cells for row in rows] if rows else []
        return format_text_response(schemas)
    except Exception as e:
        logger.error(f"Error listing schemas: {e}")
        return format_error_response(str(e))


@mcp.tool(description="List objects in a schema")
async def list_objects(
    schema_name: str = Field(description="Schema name"),
    object_type: str = Field(description="Object type: 'table', 'view', 'sequence', or 'extension'", default="table"),
) -> ResponseType:
    """List objects of a given type in a schema."""
    try:
        sql_driver = await get_sql_driver()

        if object_type in ("table", "view"):
            table_type = "BASE TABLE" if object_type == "table" else "VIEW"
            rows = await SafeSqlDriver.execute_param_query(
                sql_driver,
                """
                SELECT table_schema, table_name, table_type
                FROM information_schema.tables
                WHERE table_schema = {} AND table_type = {}
                ORDER BY table_name
                """,
                [schema_name, table_type],
            )
            objects = (
                [{"schema": row.cells["table_schema"], "name": row.cells["table_name"], "type": row.cells["table_type"]} for row in rows]
                if rows
                else []
            )

        elif object_type == "sequence":
            rows = await SafeSqlDriver.execute_param_query(
                sql_driver,
                """
                SELECT sequence_schema, sequence_name, data_type
                FROM information_schema.sequences
                WHERE sequence_schema = {}
                ORDER BY sequence_name
                """,
                [schema_name],
            )
            objects = (
                [{"schema": row.cells["sequence_schema"], "name": row.cells["sequence_name"], "data_type": row.cells["data_type"]} for row in rows]
                if rows
                else []
            )

        elif object_type == "extension":
            # Extensions are not schema-specific
            rows = await sql_driver.execute_query(
                """
                SELECT extname, extversion, extrelocatable
                FROM pg_extension
                ORDER BY extname
                """
            )
            objects = (
                [{"name": row.cells["extname"], "version": row.cells["extversion"], "relocatable": row.cells["extrelocatable"]} for row in rows]
                if rows
                else []
            )

        else:
            return format_error_response(f"Unsupported object type: {object_type}")

        return format_text_response(objects)
    except Exception as e:
        logger.error(f"Error listing objects: {e}")
        return format_error_response(str(e))


@mcp.tool(description="Show detailed information about a database object")
async def get_object_details(
    schema_name: str = Field(description="Schema name"),
    object_name: str = Field(description="Object name"),
    object_type: str = Field(description="Object type: 'table', 'view', 'sequence', or 'extension'", default="table"),
) -> ResponseType:
    """Get detailed information about a database object."""
    try:
        sql_driver = await get_sql_driver()

        if object_type in ("table", "view"):
            # Get columns
            col_rows = await SafeSqlDriver.execute_param_query(
                sql_driver,
                """
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_schema = {} AND table_name = {}
                ORDER BY ordinal_position
                """,
                [schema_name, object_name],
            )
            columns = (
                [
                    {
                        "column": r.cells["column_name"],
                        "data_type": r.cells["data_type"],
                        "is_nullable": r.cells["is_nullable"],
                        "default": r.cells["column_default"],
                    }
                    for r in col_rows
                ]
                if col_rows
                else []
            )

            # Get constraints
            con_rows = await SafeSqlDriver.execute_param_query(
                sql_driver,
                """
                SELECT tc.constraint_name, tc.constraint_type, kcu.column_name
                FROM information_schema.table_constraints AS tc
                LEFT JOIN information_schema.key_column_usage AS kcu
                  ON tc.constraint_name = kcu.constraint_name
                 AND tc.table_schema = kcu.table_schema
                WHERE tc.table_schema = {} AND tc.table_name = {}
                """,
                [schema_name, object_name],
            )

            constraints = {}
            if con_rows:
                for row in con_rows:
                    cname = row.cells["constraint_name"]
                    ctype = row.cells["constraint_type"]
                    col = row.cells["column_name"]

                    if cname not in constraints:
                        constraints[cname] = {"type": ctype, "columns": []}
                    if col:
                        constraints[cname]["columns"].append(col)

            constraints_list = [{"name": name, **data} for name, data in constraints.items()]

            # Get indexes
            idx_rows = await SafeSqlDriver.execute_param_query(
                sql_driver,
                """
                SELECT indexname, indexdef
                FROM pg_indexes
                WHERE schemaname = {} AND tablename = {}
                """,
                [schema_name, object_name],
            )

            indexes = [{"name": r.cells["indexname"], "definition": r.cells["indexdef"]} for r in idx_rows] if idx_rows else []

            result = {
                "basic": {"schema": schema_name, "name": object_name, "type": object_type},
                "columns": columns,
                "constraints": constraints_list,
                "indexes": indexes,
            }

        elif object_type == "sequence":
            rows = await SafeSqlDriver.execute_param_query(
                sql_driver,
                """
                SELECT sequence_schema, sequence_name, data_type, start_value, increment
                FROM information_schema.sequences
                WHERE sequence_schema = {} AND sequence_name = {}
                """,
                [schema_name, object_name],
            )

            if rows and rows[0]:
                row = rows[0]
                result = {
                    "schema": row.cells["sequence_schema"],
                    "name": row.cells["sequence_name"],
                    "data_type": row.cells["data_type"],
                    "start_value": row.cells["start_value"],
                    "increment": row.cells["increment"],
                }
            else:
                result = {}

        elif object_type == "extension":
            rows = await SafeSqlDriver.execute_param_query(
                sql_driver,
                """
                SELECT extname, extversion, extrelocatable
                FROM pg_extension
                WHERE extname = {}
                """,
                [object_name],
            )

            if rows and rows[0]:
                row = rows[0]
                result = {"name": row.cells["extname"], "version": row.cells["extversion"], "relocatable": row.cells["extrelocatable"]}
            else:
                result = {}

        else:
            return format_error_response(f"Unsupported object type: {object_type}")

        return format_text_response(result)
    except Exception as e:
        logger.error(f"Error getting object details: {e}")
        return format_error_response(str(e))


@mcp.tool(
    description="Get table statistics without running expensive COUNT(*) queries. "
    "Returns instant estimates from PostgreSQL system catalogs including table size, row count estimate, "
    "index sizes, and last vacuum/analyze times. SAFE for large production tables."
)
async def get_table_stats(
    schema_name: str = Field(description="Schema name (e.g., 'public', 'data_api_v2')"),
    table_name: str = Field(description="Table name"),
) -> ResponseType:
    """
    Get comprehensive table statistics without scanning the table.

    This tool queries PostgreSQL system catalogs to provide instant statistics:
    - Estimated row count (from pg_class.reltuples)
    - Table size in bytes/MB/GB
    - Index sizes
    - Toast table size (for large values)
    - Last vacuum and analyze timestamps
    - Live/dead tuple counts

    Safe for large tables in production - does NOT run COUNT(*) or table scans.
    """
    try:
        sql_driver = await get_sql_driver()

        query = """
            SELECT
                schemaname,
                tablename,
                -- Size information (instantly available)
                pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as total_size,
                pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) as table_size,
                pg_size_pretty(pg_indexes_size(schemaname||'.'||tablename)) as indexes_size,
                pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename) -
                              pg_relation_size(schemaname||'.'||tablename) -
                              pg_indexes_size(schemaname||'.'||tablename)) as toast_size,
                -- Raw sizes in bytes
                pg_total_relation_size(schemaname||'.'||tablename) as total_bytes,
                pg_relation_size(schemaname||'.'||tablename) as table_bytes,
                pg_indexes_size(schemaname||'.'||tablename) as indexes_bytes,
                -- Row count estimates from statistics
                n_live_tup as estimated_live_rows,
                n_dead_tup as estimated_dead_rows,
                n_live_tup + n_dead_tup as estimated_total_rows,
                -- Last maintenance
                last_vacuum,
                last_autovacuum,
                last_analyze,
                last_autoanalyze,
                -- Table activity
                seq_scan as sequential_scans,
                idx_scan as index_scans,
                n_tup_ins as inserts,
                n_tup_upd as updates,
                n_tup_del as deletes
            FROM pg_stat_user_tables
            WHERE schemaname = {}
              AND tablename = {}
        """

        rows = await SafeSqlDriver.execute_param_query(sql_driver, query, [schema_name, table_name])

        if not rows or len(rows) == 0:
            return format_error_response(f"Table not found: {schema_name}.{table_name}")

        stats = dict(rows[0].cells)

        # Add reltuples from pg_class for alternative estimate
        reltuples_query = """
            SELECT c.reltuples::bigint as reltuples_estimate
            FROM pg_class c
            JOIN pg_namespace n ON c.relnamespace = n.oid
            WHERE n.nspname = {}
              AND c.relname = {}
        """

        reltuples_rows = await SafeSqlDriver.execute_param_query(
            sql_driver, reltuples_query, [schema_name, table_name]
        )

        if reltuples_rows and len(reltuples_rows) > 0:
            stats['reltuples_estimate'] = reltuples_rows[0].cells.get('reltuples_estimate')

        # Add helpful notes
        result = {
            "table": f"{schema_name}.{table_name}",
            "size": {
                "total": stats.get('total_size'),
                "table": stats.get('table_size'),
                "indexes": stats.get('indexes_size'),
                "toast": stats.get('toast_size'),
                "total_bytes": stats.get('total_bytes'),
                "table_bytes": stats.get('table_bytes'),
                "indexes_bytes": stats.get('indexes_bytes'),
            },
            "rows": {
                "estimated_live": stats.get('estimated_live_rows'),
                "estimated_dead": stats.get('estimated_dead_rows'),
                "estimated_total": stats.get('estimated_total_rows'),
                "reltuples_estimate": stats.get('reltuples_estimate'),
                "note": "These are estimates from PostgreSQL statistics. "
                        "Run ANALYZE table_name to update estimates. "
                        "For exact count, use COUNT(*) but beware of performance on large tables."
            },
            "maintenance": {
                "last_vacuum": str(stats.get('last_vacuum')) if stats.get('last_vacuum') else None,
                "last_autovacuum": str(stats.get('last_autovacuum')) if stats.get('last_autovacuum') else None,
                "last_analyze": str(stats.get('last_analyze')) if stats.get('last_analyze') else None,
                "last_autoanalyze": str(stats.get('last_autoanalyze')) if stats.get('last_autoanalyze') else None,
            },
            "activity": {
                "sequential_scans": stats.get('sequential_scans'),
                "index_scans": stats.get('index_scans'),
                "inserts": stats.get('inserts'),
                "updates": stats.get('updates'),
                "deletes": stats.get('deletes'),
            }
        }

        return format_text_response(result)

    except Exception as e:
        logger.error(f"Error getting table stats: {e}")
        return format_error_response(str(e))


@mcp.tool(
    description="Estimate query cost WITHOUT running the query. "
    "Returns estimated execution time, rows, and cost from EXPLAIN (no actual execution). "
    "SAFE for expensive queries - use this before running queries on large tables."
)
async def estimate_query_cost(
    sql: str = Field(description="SQL query to estimate (SELECT, UPDATE, DELETE, INSERT)"),
) -> ResponseType:
    """
    Estimate query cost without executing it.

    Uses EXPLAIN (without ANALYZE) to get PostgreSQL's cost estimates:
    - Estimated startup cost
    - Estimated total cost
    - Estimated rows returned
    - Query plan structure

    Does NOT execute the query, so it's safe for expensive operations.
    Use this before running queries on large tables to check if they'll be slow.
    """
    try:
        sql_driver = await get_sql_driver()

        # Use EXPLAIN (not ANALYZE) to get estimates without execution
        explain_query = f"EXPLAIN (FORMAT JSON, VERBOSE) {sql}"

        rows = await sql_driver.execute_query(explain_query)

        if not rows or len(rows) == 0:
            return format_error_response("No explain plan returned")

        # Parse the JSON explain output
        plan_json = rows[0].cells.get('QUERY PLAN')

        if isinstance(plan_json, str):
            import json
            plan_json = json.loads(plan_json)

        if not plan_json or len(plan_json) == 0:
            return format_error_response("Invalid explain plan format")

        plan = plan_json[0].get('Plan', {})

        result = {
            "query": sql,
            "estimated_cost": {
                "startup_cost": plan.get('Startup Cost'),
                "total_cost": plan.get('Total Cost'),
                "note": "Cost is in arbitrary units. Higher = more expensive. "
                        "As a rule of thumb: <1000 is fast, >10000 may be slow, >100000 is very expensive."
            },
            "estimated_rows": plan.get('Plan Rows'),
            "estimated_width": plan.get('Plan Width'),
            "plan_type": plan.get('Node Type'),
            "full_plan": plan,
            "warning": "This is an ESTIMATE. Actual execution may differ. "
                      "For large tables, consider adding LIMIT or WHERE clauses to reduce cost."
        }

        # Add cost assessment
        total_cost = plan.get('Total Cost', 0)
        if total_cost < 1000:
            result['assessment'] = "FAST - This query should execute quickly"
        elif total_cost < 10000:
            result['assessment'] = "MODERATE - This query may take a few seconds"
        elif total_cost < 100000:
            result['assessment'] = "SLOW - This query may take 10+ seconds"
        else:
            result['assessment'] = "VERY EXPENSIVE - This query may take minutes or hang. Consider optimization."

        return format_text_response(result)

    except Exception as e:
        logger.error(f"Error estimating query cost: {e}")
        return format_error_response(str(e))


@mcp.tool(description="Explains the execution plan for a SQL query, showing how the database will execute it and provides detailed cost estimates.")
async def explain_query(
    sql: str = Field(description="SQL query to explain"),
    analyze: bool = Field(
        description="When True, actually runs the query to show real execution statistics instead of estimates. "
        "Takes longer but provides more accurate information.",
        default=False,
    ),
    hypothetical_indexes: list[dict[str, Any]] = Field(
        description="""A list of hypothetical indexes to simulate. Each index must be a dictionary with these keys:
    - 'table': The table name to add the index to (e.g., 'users')
    - 'columns': List of column names to include in the index (e.g., ['email'] or ['last_name', 'first_name'])
    - 'using': Optional index method (default: 'btree', other options include 'hash', 'gist', etc.)

Examples: [
    {"table": "users", "columns": ["email"], "using": "btree"},
    {"table": "orders", "columns": ["user_id", "created_at"]}
]
If there is no hypothetical index, you can pass an empty list.""",
        default=[],
    ),
) -> ResponseType:
    """
    Explains the execution plan for a SQL query.

    Args:
        sql: The SQL query to explain
        analyze: When True, actually runs the query for real statistics
        hypothetical_indexes: Optional list of indexes to simulate
    """
    try:
        sql_driver = await get_sql_driver()
        explain_tool = ExplainPlanTool(sql_driver=sql_driver)
        result: ExplainPlanArtifact | ErrorResult | None = None

        # If hypothetical indexes are specified, check for HypoPG extension
        if hypothetical_indexes and len(hypothetical_indexes) > 0:
            if analyze:
                return format_error_response("Cannot use analyze and hypothetical indexes together")
            try:
                # Use the common utility function to check if hypopg is installed
                (
                    is_hypopg_installed,
                    hypopg_message,
                ) = await check_hypopg_installation_status(sql_driver)

                # If hypopg is not installed, return the message
                if not is_hypopg_installed:
                    return format_text_response(hypopg_message)

                # HypoPG is installed, proceed with explaining with hypothetical indexes
                result = await explain_tool.explain_with_hypothetical_indexes(sql, hypothetical_indexes)
            except Exception:
                raise  # Re-raise the original exception
        elif analyze:
            try:
                # Use EXPLAIN ANALYZE
                result = await explain_tool.explain_analyze(sql)
            except Exception:
                raise  # Re-raise the original exception
        else:
            try:
                # Use basic EXPLAIN
                result = await explain_tool.explain(sql)
            except Exception:
                raise  # Re-raise the original exception

        if result and isinstance(result, ExplainPlanArtifact):
            return format_text_response(result.to_text())
        else:
            error_message = "Error processing explain plan"
            if isinstance(result, ErrorResult):
                error_message = result.to_text()
            return format_error_response(error_message)
    except Exception as e:
        logger.error(f"Error explaining query: {e}")
        return format_error_response(str(e))


# Query function declaration without the decorator - we'll add it dynamically based on access mode
async def execute_sql(
    sql: str = Field(description="SQL to run", default="all"),
) -> ResponseType:
    """Executes a SQL query against the database."""
    try:
        sql_driver = await get_sql_driver()
        rows = await sql_driver.execute_query(sql)  # type: ignore
        if rows is None:
            return format_text_response("No results")
        return format_text_response(list([r.cells for r in rows]))
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        return format_error_response(str(e))


@mcp.tool(description="Analyze frequently executed queries in the database and recommend optimal indexes")
@validate_call
async def analyze_workload_indexes(
    max_index_size_mb: int = Field(description="Max index size in MB", default=10000),
    method: Literal["dta", "llm"] = Field(description="Method to use for analysis", default="dta"),
) -> ResponseType:
    """Analyze frequently executed queries in the database and recommend optimal indexes."""
    try:
        sql_driver = await get_sql_driver()
        if method == "dta":
            index_tuning = DatabaseTuningAdvisor(sql_driver)
        else:
            index_tuning = LLMOptimizerTool(sql_driver)
        dta_tool = TextPresentation(sql_driver, index_tuning)
        result = await dta_tool.analyze_workload(max_index_size_mb=max_index_size_mb)
        return format_text_response(result)
    except Exception as e:
        logger.error(f"Error analyzing workload: {e}")
        return format_error_response(str(e))


@mcp.tool(description="Analyze a list of (up to 10) SQL queries and recommend optimal indexes")
@validate_call
async def analyze_query_indexes(
    queries: list[str] = Field(description="List of Query strings to analyze"),
    max_index_size_mb: int = Field(description="Max index size in MB", default=10000),
    method: Literal["dta", "llm"] = Field(description="Method to use for analysis", default="dta"),
) -> ResponseType:
    """Analyze a list of SQL queries and recommend optimal indexes."""
    if len(queries) == 0:
        return format_error_response("Please provide a non-empty list of queries to analyze.")
    if len(queries) > MAX_NUM_INDEX_TUNING_QUERIES:
        return format_error_response(f"Please provide a list of up to {MAX_NUM_INDEX_TUNING_QUERIES} queries to analyze.")

    try:
        sql_driver = await get_sql_driver()
        if method == "dta":
            index_tuning = DatabaseTuningAdvisor(sql_driver)
        else:
            index_tuning = LLMOptimizerTool(sql_driver)
        dta_tool = TextPresentation(sql_driver, index_tuning)
        result = await dta_tool.analyze_queries(queries=queries, max_index_size_mb=max_index_size_mb)
        return format_text_response(result)
    except Exception as e:
        logger.error(f"Error analyzing queries: {e}")
        return format_error_response(str(e))


@mcp.tool(
    description="Analyzes database health. Here are the available health checks:\n"
    "- index - checks for invalid, duplicate, and bloated indexes\n"
    "- connection - checks the number of connection and their utilization\n"
    "- vacuum - checks vacuum health for transaction id wraparound\n"
    "- sequence - checks sequences at risk of exceeding their maximum value\n"
    "- replication - checks replication health including lag and slots\n"
    "- buffer - checks for buffer cache hit rates for indexes and tables\n"
    "- constraint - checks for invalid constraints\n"
    "- all - runs all checks\n"
    "You can optionally specify a single health check or a comma-separated list of health checks. The default is 'all' checks."
)
async def analyze_db_health(
    health_type: str = Field(
        description=f"Optional. Valid values are: {', '.join(sorted([t.value for t in HealthType]))}.",
        default="all",
    ),
) -> ResponseType:
    """Analyze database health for specified components.

    Args:
        health_type: Comma-separated list of health check types to perform.
                    Valid values: index, connection, vacuum, sequence, replication, buffer, constraint, all
    """
    health_tool = DatabaseHealthTool(await get_sql_driver())
    result = await health_tool.health(health_type=health_type)
    return format_text_response(result)


@mcp.tool(
    name="get_top_queries",
    description=f"Reports the slowest or most resource-intensive queries using data from the '{PG_STAT_STATEMENTS}' extension.",
)
async def get_top_queries(
    sort_by: str = Field(
        description="Ranking criteria: 'total_time' for total execution time or 'mean_time' for mean execution time per call, or 'resources' "
        "for resource-intensive queries",
        default="resources",
    ),
    limit: int = Field(description="Number of queries to return when ranking based on mean_time or total_time", default=10),
) -> ResponseType:
    try:
        sql_driver = await get_sql_driver()
        top_queries_tool = TopQueriesCalc(sql_driver=sql_driver)

        if sort_by == "resources":
            result = await top_queries_tool.get_top_resource_queries()
            return format_text_response(result)
        elif sort_by == "mean_time" or sort_by == "total_time":
            # Map the sort_by values to what get_top_queries_by_time expects
            result = await top_queries_tool.get_top_queries_by_time(limit=limit, sort_by="mean" if sort_by == "mean_time" else "total")
        else:
            return format_error_response("Invalid sort criteria. Please use 'resources' or 'mean_time' or 'total_time'.")
        return format_text_response(result)
    except Exception as e:
        logger.error(f"Error getting slow queries: {e}")
        return format_error_response(str(e))


# DAPI Tools - for querying DAPI (Data API) historical data
@mcp.tool(
    description="Query DAPI historical data with pagination. Returns entities of a given type (component_name) "
    "with filters for organization, business, agency, and time range. Supports pagination via offset/limit."
)
async def dapi_query_historical_data(
    component_name: str = Field(description="Entity type (e.g., 'contact', 'listing', 'agent', 'offer')"),
    org_name: str = Field(description="Organization name"),
    business_id: str | None = Field(description="Optional business UUID filter", default=None),
    agency_id: str | None = Field(description="Optional agency UUID filter", default=None),
    offset: int = Field(description="Pagination offset (row_number to start from)", default=0),
    limit: int = Field(description="Maximum number of records to return (1-1000)", default=100),
    since: str | None = Field(description="Optional ISO timestamp - only return entities modified after this time", default=None),
    include_deleted: bool = Field(description="Include soft-deleted entities", default=False),
) -> ResponseType:
    """Query DAPI historical data with pagination and filtering."""
    return await dapi_tools.query_historical_data(
        component_name=component_name,
        org_name=org_name,
        business_id=business_id,
        agency_id=agency_id,
        offset=offset,
        limit=limit,
        since=since,
        include_deleted=include_deleted,
        sql_driver=await get_sql_driver(),
    )


@mcp.tool(description="Fetch a single DAPI entity by its ID. Fast lookup when you know the entity's UUID.")
async def dapi_fetch_entity_by_id(
    component_name: str = Field(description="Entity type (e.g., 'contact', 'listing', 'agent')"),
    org_name: str = Field(description="Organization name"),
    entity_id: str = Field(description="Entity UUID"),
) -> ResponseType:
    """Fetch a single DAPI entity by ID."""
    return await dapi_tools.fetch_entity_by_id(
        component_name=component_name,
        org_name=org_name,
        entity_id=entity_id,
        sql_driver=await get_sql_driver(),
    )


@mcp.tool(
    description="Extract a specific field from a DAPI entity's JSON payload using dot notation. "
    "Useful for navigating nested JSON without fetching the entire entity. "
    "Example: field_path='data.firstName' or 'email'"
)
async def dapi_extract_payload_field(
    component_name: str = Field(description="Entity type (e.g., 'contact', 'listing')"),
    org_name: str = Field(description="Organization name"),
    entity_id: str = Field(description="Entity UUID"),
    field_path: str = Field(description="Dot-notation path to field in payload (e.g., 'data.firstName')"),
) -> ResponseType:
    """Extract a specific field from an entity's JSON payload."""
    return await dapi_tools.extract_payload_field(
        component_name=component_name,
        org_name=org_name,
        entity_id=entity_id,
        field_path=field_path,
        sql_driver=await get_sql_driver(),
    )


@mcp.tool(
    description="Find DAPI entities by their external correlation ID. "
    "Tracks entities that originated from external systems (e.g., REA, Domain) by their external source ID."
)
async def dapi_query_by_correlation_id(
    component_name: str = Field(description="Entity type (e.g., 'contact', 'listing')"),
    org_name: str = Field(description="Organization name"),
    source: str = Field(description="External source system (e.g., 'REA', 'Domain')"),
    correlation_id: str = Field(description="External correlation/source ID"),
) -> ResponseType:
    """Find entities by external correlation ID."""
    return await dapi_tools.query_by_correlation_id(
        component_name=component_name,
        org_name=org_name,
        source=source,
        correlation_id=correlation_id,
        sql_driver=await get_sql_driver(),
    )


@mcp.tool(
    description="List all available DAPI component types (entity types) for an organization. "
    "Returns counts and last modified dates for each component type."
)
async def dapi_list_component_names(
    org_name: str = Field(description="Organization name"),
) -> ResponseType:
    """List all available DAPI component types."""
    return await dapi_tools.list_component_names(
        org_name=org_name,
        sql_driver=await get_sql_driver(),
    )


@mcp.tool(
    description="Query deleted entities from DAPI. "
    "Track deletions and merges. When entities are merged, returns the target entity information."
)
async def dapi_query_deleted_entities(
    component_name: str = Field(description="Entity type (e.g., 'contact', 'listing')"),
    org_name: str = Field(description="Organization name"),
    business_id: str | None = Field(description="Optional business UUID filter", default=None),
    since: str | None = Field(description="Optional ISO timestamp - only return entities deleted after this time", default=None),
    include_merge_targets: bool = Field(description="Include merge_payload for merged entities", default=True),
    limit: int = Field(description="Maximum number of records to return (1-1000)", default=100),
) -> ResponseType:
    """Query deleted DAPI entities."""
    return await dapi_tools.query_deleted_entities(
        component_name=component_name,
        org_name=org_name,
        business_id=business_id,
        since=since,
        include_merge_targets=include_merge_targets,
        limit=limit,
        sql_driver=await get_sql_driver(),
    )


@mcp.tool(
    description="Fetch multiple DAPI entities by their IDs in a single query. "
    "Efficient bulk fetch (max 100 IDs) that reduces round trips."
)
async def dapi_batch_fetch_by_ids(
    component_name: str = Field(description="Entity type (e.g., 'contact', 'listing')"),
    org_name: str = Field(description="Organization name"),
    entity_ids: list[str] = Field(description="List of entity UUIDs (max 100)"),
) -> ResponseType:
    """Batch fetch multiple entities by IDs."""
    return await dapi_tools.batch_fetch_by_ids(
        component_name=component_name,
        org_name=org_name,
        entity_ids=entity_ids,
        sql_driver=await get_sql_driver(),
    )


@mcp.tool(
    description="Get count statistics for DAPI entities. "
    "Returns total, active, deleted, and merged counts for a component type."
)
async def dapi_count_entities(
    component_name: str = Field(description="Entity type (e.g., 'contact', 'listing')"),
    org_name: str = Field(description="Organization name"),
    business_id: str | None = Field(description="Optional business UUID filter", default=None),
    include_deleted: bool = Field(description="Include deleted entities in count", default=False),
) -> ResponseType:
    """Count DAPI entities with statistics."""
    return await dapi_tools.count_entities(
        component_name=component_name,
        org_name=org_name,
        business_id=business_id,
        include_deleted=include_deleted,
        sql_driver=await get_sql_driver(),
    )


# ============================================================================
# MONOLITH POSTGRESQL TOOLS
# These tools query operational tables in the monolith PostgreSQL database
# (separate from DAPI). Use these for user lookups, agent-contact relationships,
# and entity transfer tracking.
# ============================================================================


@mcp.tool(
    description="[MONOLITH] Fetch a single user by ID from fts_users table. "
    "Returns complete user record including email, name, business, and timestamps."
)
async def monolith_fetch_user_by_id(
    user_id: str = Field(description="User UUID"),
) -> ResponseType:
    """Fetch a user by ID from monolith fts_users table."""
    return await monolith_tools.fetch_user_by_id(
        user_id=user_id,
        sql_driver=await get_sql_driver(),
    )


@mcp.tool(
    description="[MONOLITH] Fetch a user by email from fts_users table. "
    "Useful for authentication, user identification, and support queries."
)
async def monolith_fetch_user_by_email(
    email: str = Field(description="User email address"),
) -> ResponseType:
    """Fetch a user by email from monolith fts_users table."""
    return await monolith_tools.fetch_user_by_email(
        email=email,
        sql_driver=await get_sql_driver(),
    )


@mcp.tool(
    description="[MONOLITH] Search users from fts_users with filters and pagination. "
    "Supports text search (name/email), business filter, archived status, and internal user filtering."
)
async def monolith_fetch_users(
    business_fk: str | None = Field(description="Optional business UUID filter", default=None),
    text_search: str | None = Field(description="Optional text search (name or email)", default=None),
    archived: bool | None = Field(description="Optional filter by archived status", default=None),
    is_internal: bool | None = Field(description="Optional filter for internal users (staff/agents)", default=None),
    limit: int = Field(description="Maximum results to return (1-1000)", default=100),
    offset: int = Field(description="Pagination offset", default=0),
) -> ResponseType:
    """Search users from monolith fts_users table."""
    return await monolith_tools.fetch_users(
        business_fk=business_fk,
        text_search=text_search,
        archived=archived,
        is_internal=is_internal,
        limit=limit,
        offset=offset,
        sql_driver=await get_sql_driver(),
    )


@mcp.tool(
    description="[MONOLITH] Fetch agent-contact relationships by agent ID. "
    "Traverses from agent → contacts. Use to find all contacts associated with an agent."
)
async def monolith_fetch_acr_by_agent_id(
    agent_id: str = Field(description="Agent UUID"),
    limit: int = Field(description="Maximum results to return (1-1000)", default=100),
    offset: int = Field(description="Pagination offset", default=0),
) -> ResponseType:
    """Fetch ACR by agent_id from monolith agent_contact_relationship table."""
    return await monolith_tools.fetch_acr_by_agent_id(
        agent_id=agent_id,
        limit=limit,
        offset=offset,
        sql_driver=await get_sql_driver(),
    )


@mcp.tool(
    description="[MONOLITH] Fetch agent-contact relationships by contact ID. "
    "Traverses from contact → agents. Use to find all agents associated with a contact. "
    "Bidirectional with monolith_fetch_acr_by_agent_id."
)
async def monolith_fetch_acr_by_contact_id(
    contact_id: str = Field(description="Contact UUID"),
    limit: int = Field(description="Maximum results to return (1-1000)", default=100),
    offset: int = Field(description="Pagination offset", default=0),
) -> ResponseType:
    """Fetch ACR by contact_id from monolith agent_contact_relationship table."""
    return await monolith_tools.fetch_acr_by_contact_id(
        contact_id=contact_id,
        limit=limit,
        offset=offset,
        sql_driver=await get_sql_driver(),
    )


@mcp.tool(
    description="[MONOLITH] Fetch entity transfers from a specific source entity by a specific agent. "
    "Query: 'What did Agent X transfer from Entity Y?' Tracks ownership changes in the audit trail."
)
async def monolith_fetch_atel_by_from_entity_id_and_agent_id(
    from_entity_id: str = Field(description="Source entity UUID (transferred FROM)"),
    agent_id: str = Field(description="Agent UUID involved in transfer"),
    limit: int = Field(description="Maximum results to return (1-1000)", default=100),
    offset: int = Field(description="Pagination offset", default=0),
) -> ResponseType:
    """Fetch ATEL by from_entity_id and agent_id from agent_transfer_entity_ledger."""
    return await monolith_tools.fetch_atel_by_from_entity_id_and_agent_id(
        from_entity_id=from_entity_id,
        agent_id=agent_id,
        limit=limit,
        offset=offset,
        sql_driver=await get_sql_driver(),
    )


@mcp.tool(
    description="[MONOLITH] Fetch all transfers FROM a specific source entity. "
    "Complete transfer history showing who received the entity and when. Use for audit trail."
)
async def monolith_fetch_atel_by_from_entity_id(
    from_entity_id: str = Field(description="Source entity UUID (transferred FROM)"),
    limit: int = Field(description="Maximum results to return (1-1000)", default=100),
    offset: int = Field(description="Pagination offset", default=0),
) -> ResponseType:
    """Fetch ATEL by from_entity_id from agent_transfer_entity_ledger."""
    return await monolith_tools.fetch_atel_by_from_entity_id(
        from_entity_id=from_entity_id,
        limit=limit,
        offset=offset,
        sql_driver=await get_sql_driver(),
    )


@mcp.tool(
    description="[MONOLITH] Fetch all transfers TO a specific destination entity. "
    "Shows what entities were transferred into an agent/entity's portfolio. "
    "Bidirectional with monolith_fetch_atel_by_from_entity_id."
)
async def monolith_fetch_atel_by_to_entity_id(
    to_entity_id: str = Field(description="Destination entity UUID (receiving transfers)"),
    limit: int = Field(description="Maximum results to return (1-1000)", default=100),
    offset: int = Field(description="Pagination offset", default=0),
) -> ResponseType:
    """Fetch ATEL by to_entity_id from agent_transfer_entity_ledger."""
    return await monolith_tools.fetch_atel_by_to_entity_id(
        to_entity_id=to_entity_id,
        limit=limit,
        offset=offset,
        sql_driver=await get_sql_driver(),
    )


@mcp.tool(
    description="[MONOLITH] Fetch a single property by ID from fts_properties table. "
    "Returns complete property record including address, agents, vendor info, phase, lead scoring, etc."
)
async def monolith_fetch_property_by_id(
    property_id: str = Field(description="Property UUID"),
) -> ResponseType:
    """Fetch a property by ID from monolith fts_properties table."""
    return await monolith_tools.fetch_property_by_id(
        property_id=property_id,
        sql_driver=await get_sql_driver(),
    )


@mcp.tool(
    description="[MONOLITH] Search properties from fts_properties with filters and pagination. "
    "Supports business filter and address search. Results ordered by lead priority."
)
async def monolith_fetch_properties(
    business_fk: str | None = Field(description="Optional business UUID filter", default=None),
    address: str | None = Field(description="Optional text search for address", default=None),
    limit: int = Field(description="Maximum results to return (1-1000)", default=100),
    offset: int = Field(description="Pagination offset", default=0),
) -> ResponseType:
    """Search properties from monolith fts_properties table."""
    return await monolith_tools.fetch_properties(
        business_fk=business_fk,
        address=address,
        limit=limit,
        offset=offset,
        sql_driver=await get_sql_driver(),
    )


@mcp.tool(
    description="[MONOLITH] Fetch a single agent by ID from fts_users table (where is_internal = true). "
    "Agents are staff/internal users distinguished from contacts/customers."
)
async def monolith_fetch_agent_by_id(
    agent_id: str = Field(description="Agent UUID"),
) -> ResponseType:
    """Fetch an agent by ID from monolith fts_users table."""
    return await monolith_tools.fetch_agent_by_id(
        agent_id=agent_id,
        sql_driver=await get_sql_driver(),
    )


@mcp.tool(
    description="[MONOLITH] Fetch an agent by email from fts_users table (where is_internal = true). "
    "Only returns internal staff/agents, not contacts/customers."
)
async def monolith_fetch_agent_by_email(
    email: str = Field(description="Agent email address"),
) -> ResponseType:
    """Fetch an agent by email from monolith fts_users table."""
    return await monolith_tools.fetch_agent_by_email(
        email=email,
        sql_driver=await get_sql_driver(),
    )


@mcp.tool(
    description="[MONOLITH] Fetch agents from fts_users with optional business filter and pagination. "
    "Only returns internal staff/agents (where is_internal = true). Results ordered by org_name and full_name."
)
async def monolith_fetch_agents(
    business_id: str | None = Field(description="Optional business UUID filter", default=None),
    limit: int = Field(description="Maximum results to return (1-1000)", default=100),
    offset: int = Field(description="Pagination offset", default=0),
) -> ResponseType:
    """Fetch agents from monolith fts_users table."""
    return await monolith_tools.fetch_agents(
        business_id=business_id,
        limit=limit,
        offset=offset,
        sql_driver=await get_sql_driver(),
    )


async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="PostgreSQL MCP Server")
    parser.add_argument("database_url", help="Database connection URL", nargs="?")
    parser.add_argument(
        "--access-mode",
        type=str,
        choices=[mode.value for mode in AccessMode],
        default=AccessMode.UNRESTRICTED.value,
        help="Set SQL access mode: unrestricted (unrestricted) or restricted (read-only with protections)",
    )
    parser.add_argument(
        "--transport",
        type=str,
        choices=["stdio", "sse"],
        default="stdio",
        help="Select MCP transport: stdio (default) or sse",
    )
    parser.add_argument(
        "--sse-host",
        type=str,
        default="localhost",
        help="Host to bind SSE server to (default: localhost)",
    )
    parser.add_argument(
        "--sse-port",
        type=int,
        default=8000,
        help="Port for SSE server (default: 8000)",
    )

    args = parser.parse_args()

    # Store the access mode in the global variable
    global current_access_mode
    current_access_mode = AccessMode(args.access_mode)

    # Add the query tool with a description appropriate to the access mode
    if current_access_mode == AccessMode.UNRESTRICTED:
        mcp.add_tool(execute_sql, description=(
            "Execute any SQL query. "
            "⚠️ WARNING: Avoid COUNT(*) on large tables (use get_table_stats instead). "
            "Use estimate_query_cost to check query cost before running expensive queries on production databases."
        ))
    else:
        mcp.add_tool(execute_sql, description=(
            "Execute a read-only SQL query. "
            "⚠️ WARNING: Avoid COUNT(*) on large tables (use get_table_stats instead). "
            "Use estimate_query_cost to check query cost before running expensive queries. "
            "Has 30-second timeout for safety."
        ))

    logger.info(f"Starting PostgreSQL MCP Server in {current_access_mode.upper()} mode")

    # Get database URL from environment variable or command line
    database_url = os.environ.get("DATABASE_URI", args.database_url)

    if not database_url:
        raise ValueError(
            "Error: No database URL provided. Please specify via 'DATABASE_URI' environment variable or command-line argument.",
        )

    # Initialize database connection pool
    try:
        await db_connection.pool_connect(database_url)
        logger.info("Successfully connected to database and initialized connection pool")
    except Exception as e:
        logger.warning(
            f"Could not connect to database: {obfuscate_password(str(e))}",
        )
        logger.warning(
            "The MCP server will start but database operations will fail until a valid connection is established.",
        )

    # Set up proper shutdown handling
    try:
        loop = asyncio.get_running_loop()
        signals = (signal.SIGTERM, signal.SIGINT)
        for s in signals:
            loop.add_signal_handler(s, lambda s=s: asyncio.create_task(shutdown(s)))
    except NotImplementedError:
        # Windows doesn't support signals properly
        logger.warning("Signal handling not supported on Windows")
        pass

    # Run the server with the selected transport (always async)
    if args.transport == "stdio":
        await mcp.run_stdio_async()
    else:
        # Update FastMCP settings based on command line arguments
        mcp.settings.host = args.sse_host
        mcp.settings.port = args.sse_port
        await mcp.run_sse_async()


async def shutdown(sig=None):
    """Clean shutdown of the server."""
    global shutdown_in_progress

    if shutdown_in_progress:
        logger.warning("Forcing immediate exit")
        # Use sys.exit instead of os._exit to allow for proper cleanup
        sys.exit(1)

    shutdown_in_progress = True

    if sig:
        logger.info(f"Received exit signal {sig.name}")

    # Close database connections
    try:
        await db_connection.close()
        logger.info("Closed database connections")
    except Exception as e:
        logger.error(f"Error closing database connections: {e}")

    # Exit with appropriate status code
    sys.exit(128 + sig if sig is not None else 0)
