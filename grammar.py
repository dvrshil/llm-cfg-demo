clickhouse_flights_grammar = r"""
// Minimal, readable grammar for our flights demo (whitespace is ignored).

%import common.WS
%import common.CNAME       -> IDENT
%import common.INT         -> NUMBER
%import common.FLOAT       -> FLOAT
%ignore WS

start: "SELECT" select_list "FROM" table where? group_by? order_by? limit? ";"

table: "flights_df"

// ---- Projections ----
select_list: select_item ("," select_item)*
select_item: projection ("AS" alias)?

projection: column
          | agg_func "(" measure ")"
          | "count" "(" "*" ")"
          | "countIf" "(" condition ")"
          | mph_expr
          | dep_hour_expr
          | arr_hour_expr

agg_func: "avg" | "sum" | "min" | "max"
measure: numeric_field | mph_expr

// ---- Columns ----
column: "AIRLINE" | "ORIGIN_AIRPORT" | "DESTINATION_AIRPORT" | "DAY_OF_WEEK" | "FLIGHT_DATE"

numeric_field: "DEPARTURE_DELAY" | "ARRIVAL_DELAY" | "DISTANCE" | "AIR_TIME" | "ELAPSED_TIME" | "SCHEDULED_TIME" | "DAY_OF_WEEK"

// ---- Derived helpers ----
mph_expr: "(" "DISTANCE" "*" "60" "/" "AIR_TIME" ")"
dep_hour_expr: "intDiv" "(" "SCHEDULED_DEPARTURE" "," "100" ")"
arr_hour_expr: "intDiv" "(" "SCHEDULED_ARRIVAL" "," "100" ")"

// ---- WHERE (AND-only) ----
where: "WHERE" boolean_conj
boolean_conj: condition ("AND" condition)*

condition: comparison | date_between | hour_between_dep | hour_between_arr | airline_eq | origin_eq | destination_eq | earlylate_clause

comparison: numeric_field comp_op number
comp_op: "=" | ">" | "<" | ">=" | "<="

date_between: "FLIGHT_DATE" "BETWEEN" DATE "AND" DATE
hour_between_dep: dep_hour_expr "BETWEEN" hour "AND" hour
hour_between_arr: arr_hour_expr "BETWEEN" hour "AND" hour

airline_eq: "AIRLINE" "=" UPPER2
origin_eq: "ORIGIN_AIRPORT" "=" UPPER3
destination_eq: "DESTINATION_AIRPORT" "=" UPPER3

earlylate_clause: "abs" "(" "ARRIVAL_DELAY" ")" "<=" number
                | "abs" "(" "DEPARTURE_DELAY" ")" "<=" number
                | "ARRIVAL_DELAY" ">=" number
                | "DEPARTURE_DELAY" ">=" number

// ---- GROUP BY / ORDER BY / LIMIT ----
group_by: "GROUP" "BY" group_cols
group_cols: group_expr ("," group_expr)*
group_expr: column | dep_hour_expr | arr_hour_expr

order_by: "ORDER" "BY" order_cols
order_cols: order_expr ("," order_expr)*
order_expr: (agg_func "(" measure ")" | column | dep_hour_expr | arr_hour_expr | mph_expr) (order_dir)?
order_dir: "ASC" | "DESC"

limit: "LIMIT" NUMBER

// ---- Terminals ----
alias: IDENT
number: NUMBER | FLOAT
DATE: /'[0-9]{4}-[0-9]{2}-[0-9]{2}'/
hour: /([01]?[0-9]|2[0-3])/
UPPER2: /'[A-Z0-9]{2}'/
UPPER3: /'[A-Z0-9]{3}'/
"""

sql_lark_cfg_tool = {
    "type": "custom",
    "name": "clickhouse_sql_grammar",
    "description": (
        "SELECT-only ClickHouse SQL over flights_df. "
        "Supports WHERE (AND only), GROUP BY, ORDER BY, LIMIT. "
        "Columns: AIRLINE, ORIGIN_AIRPORT, DESTINATION_AIRPORT, FLIGHT_DATE, DAY_OF_WEEK, "
        "DEPARTURE_DELAY, ARRIVAL_DELAY, DISTANCE, AIR_TIME, ELAPSED_TIME, SCHEDULED_TIME. "
        "Helpers: intDiv(SCHEDULED_DEPARTURE,100), intDiv(SCHEDULED_ARRIVAL,100), (DISTANCE*60/AIR_TIME). "
        "Aggregates: avg,sum,min,max,count,countIf. No joins/subqueries."
    ),
    "format": {
        "type": "grammar",
        "syntax": "lark",
        "definition": clickhouse_flights_grammar,
    },
}

