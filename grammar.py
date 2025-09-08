clickhouse_flights_grammar = r"""
// ====== Punctuation & tokens (thread spaces explicitly) ======
SP: " "
COMMA: ","
SEMI: ";"
STAR: "*"

// ====== Start ======
start: "SELECT" SP select_list SP "FROM" SP table [SP opt_where] [SP opt_group_by] [SP opt_order_by] [SP opt_limit] SEMI

// ====== Table (lock to flights_df for safety) ======
table: "flights_df"

// ====== Projections ======
select_list: select_item (COMMA SP select_item)*
select_item: projection [SP "AS" SP alias]

projection: column
          | agg_func "(" measure ")"
          | quantile_fn "(" quant_p ")" "(" measure ")"
          | "count" "(" STAR ")"
          | "countIf" "(" condition ")"
          | mph_expr
          | dep_hour_expr
          | arr_hour_expr

agg_func: "avg" | "sum" | "min" | "max"
quantile_fn: "quantileTDigest" | "quantileExact"
quant_p: /0\.[0-9]{1,2}/             // 0.1 .. 0.99

// what counts as a numeric measure
measure: numeric_field | mph_expr

// ====== Columns you expose directly ======
column: "AIRLINE"
      | "ORIGIN_AIRPORT"
      | "DESTINATION_AIRPORT"
      | "DAY_OF_WEEK"
      | "FLIGHT_DATE"

// numeric fields safe for math/agg
numeric_field: "DEPARTURE_DELAY"
             | "ARRIVAL_DELAY"
             | "DISTANCE"
             | "AIR_TIME"
             | "ELAPSED_TIME"
             | "SCHEDULED_TIME"
             | "DAY_OF_WEEK"      // allow numeric filtering by weekday (1..7)

// ====== Canonical derived expressions ======
mph_expr: "(" "DISTANCE" SP "*" SP "60" SP "/" SP "AIR_TIME" ")"
dep_hour_expr: "intDiv" "(" "SCHEDULED_DEPARTURE" COMMA SP "100" ")"
arr_hour_expr: "intDiv" "(" "SCHEDULED_ARRIVAL" COMMA SP "100" ")"

// ====== WHERE (only AND, no OR, to keep it robust) ======
opt_where: "WHERE" SP boolean_conj
boolean_conj: condition (SP "AND" SP condition)*

condition: comparison
         | date_between
         | hour_between_dep
         | hour_between_arr
         | airline_eq
         | origin_eq
         | destination_eq
         | earlylate_clause     // e.g., abs(ARRIVAL_DELAY) <= 5

comparison: numeric_field SP comp_op SP number
comp_op: "=" | ">" | "<" | ">=" | "<="

date_between: "FLIGHT_DATE" SP "BETWEEN" SP DATE SP "AND" SP DATE

hour_between_dep: dep_hour_expr SP "BETWEEN" SP hour SP "AND" SP hour
hour_between_arr: arr_hour_expr SP "BETWEEN" SP hour SP "AND" SP hour

airline_eq: "AIRLINE" SP "=" SP UPPER2        // e.g., 'AA'
origin_eq: "ORIGIN_AIRPORT" SP "=" SP UPPER3  // e.g., 'SFO'
destination_eq: "DESTINATION_AIRPORT" SP "=" SP UPPER3

earlylate_clause: "abs" "(" "ARRIVAL_DELAY" ")" SP "<=" SP number
                | "abs" "(" "DEPARTURE_DELAY" ")" SP "<=" SP number
                | "ARRIVAL_DELAY" SP ">=" SP number
                | "DEPARTURE_DELAY" SP ">=" SP number

// ====== GROUP BY ======
opt_group_by: "GROUP" SP "BY" SP group_cols
group_cols: group_expr (COMMA SP group_expr)*
group_expr: column | dep_hour_expr | arr_hour_expr

// ====== ORDER BY ======
opt_order_by: "ORDER" SP "BY" SP order_cols
order_cols: order_expr (COMMA SP order_expr)*
order_expr: (agg_func "(" measure ")" 
            | quantile_fn "(" quant_p ")" "(" measure ")"
            | column 
            | dep_hour_expr 
            | arr_hour_expr 
            | mph_expr) [SP order_dir]
order_dir: "ASC" | "DESC"

// ====== LIMIT ======
opt_limit: "LIMIT" SP NUMBER

// ====== Terminals ======
alias: IDENTIFIER
IDENTIFIER: /[A-Za-z_][A-Za-z0-9_]*/
NUMBER: /[0-9]+/
FLOAT: /[0-9]+\.[0-9]+/
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
        "Read-only ClickHouse SQL over table flights_df. "
        "Allowed: SELECT, FROM, WHERE (AND only), GROUP BY, ORDER BY, LIMIT. "
        "Columns: AIRLINE, ORIGIN_AIRPORT, DESTINATION_AIRPORT, FLIGHT_DATE, DAY_OF_WEEK, "
        "DEPARTURE_DELAY, ARRIVAL_DELAY, DISTANCE, AIR_TIME, ELAPSED_TIME, SCHEDULED_TIME, "
        "SCHEDULED_DEPARTURE, SCHEDULED_ARRIVAL (the latter two are used via intDiv for hour). "
        "Units: delays/times are minutes; distances are miles; SCHEDULED_* are HHMM integers; DAY_OF_WEEK is 1..7. "
        "Derived: intDiv(SCHEDULED_DEPARTURE,100), intDiv(SCHEDULED_ARRIVAL,100), (DISTANCE*60/AIR_TIME). "
        "Aggregates: avg,sum,min,max,count,countIf(cond),quantileTDigest/Exact(p)(field). "
        "Adhere strictly to the grammar; no joins, subqueries, or DML."
    ),
    "format": {
        "type": "grammar",
        "syntax": "lark",
        "definition": clickhouse_flights_grammar,
    },
}
