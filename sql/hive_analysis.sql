-- Example Hive SQL. Adjust the LOCATION to a path visible to Hive.
-- If running Hive in a cluster, copy the CSVs into HDFS and update LOCATION.

CREATE DATABASE IF NOT EXISTS supply_chain;
USE supply_chain;

CREATE EXTERNAL TABLE IF NOT EXISTS shipments (
  shipment_id STRING,
  supplier_id STRING,
  product_id STRING,
  warehouse_id STRING,
  route_id STRING,
  source STRING,
  destination STRING,
  shipment_date STRING,
  transport_mode STRING,
  distance_km INT,
  traffic_level STRING,
  weather_condition STRING,
  inventory_level INT,
  demand_units INT,
  expected_days INT,
  actual_days INT,
  delivery_delay_days INT,
  is_delayed INT,
  defective_product INT,
  shipping_cost DOUBLE,
  month INT
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES ("separatorChar"=",", "quoteChar"="\"", "escapeChar"="\\")
STORED AS TEXTFILE
LOCATION '/supply_chain_risk/data/shipments';

-- Example analytics: supplier delay rate
SELECT supplier_id,
       COUNT(*) AS shipment_count,
       ROUND(AVG(is_delayed) * 100, 2) AS delay_rate_pct,
       ROUND(AVG(delivery_delay_days), 2) AS avg_delay_days
FROM shipments
GROUP BY supplier_id
ORDER BY delay_rate_pct DESC;

-- Example analytics: risky routes
SELECT route_id,
       COUNT(*) AS shipment_count,
       ROUND(AVG(is_delayed) * 100, 2) AS delay_rate_pct
FROM shipments
GROUP BY route_id
ORDER BY delay_rate_pct DESC;
