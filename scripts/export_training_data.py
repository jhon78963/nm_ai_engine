#!/usr/bin/env python3
"""
Exporta historial de ventas y snapshot de precios desde PostgreSQL (nm_db).

Genera:
  - data/demand_daily.csv   → series diarias por product_id (Prophet)
  - data/price_training.csv → features de precio por producto (Ridge)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common import (
    DEMAND_DAILY_FILE,
    PRICE_TRAINING_FILE,
    ensure_directories,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEMAND_QUERY = """
SELECT
    sd.product_id,
    DATE(s.creation_time) AS sale_date,
    SUM(sd.quantity)::integer AS quantity
FROM sale_details sd
INNER JOIN sales s ON s.id = sd.sale_id
WHERE s.status = 'COMPLETED'
  AND s.is_deleted = false
GROUP BY sd.product_id, DATE(s.creation_time)
ORDER BY sd.product_id, sale_date
"""

PRICE_QUERY = """
WITH primary_size AS (
    SELECT DISTINCT ON (product_id)
        product_id,
        purchase_price,
        sale_price
    FROM product_size
    ORDER BY product_id, id
),
sales_30d AS (
    SELECT
        sd.product_id,
        COALESCE(SUM(sd.quantity), 0)::integer AS sales_last_month
    FROM sale_details sd
    INNER JOIN sales s ON s.id = sd.sale_id
    WHERE s.status = 'COMPLETED'
      AND s.is_deleted = false
      AND s.creation_time >= NOW() - INTERVAL '30 days'
    GROUP BY sd.product_id
),
sales_all AS (
    SELECT
        sd.product_id,
        COALESCE(SUM(sd.quantity), 0)::integer AS total_sales_all_time,
        MAX(s.creation_time) AS last_sale_at
    FROM sale_details sd
    INNER JOIN sales s ON s.id = sd.sale_id
    WHERE s.status = 'COMPLETED'
      AND s.is_deleted = false
    GROUP BY sd.product_id
),
stock_totals AS (
    SELECT
        product_id,
        COALESCE(SUM(quantity), 0)::integer AS current_stock
    FROM inventory_balances
    WHERE color_id IS NULL
    GROUP BY product_id
)
SELECT
    p.id AS product_id,
    ps.purchase_price::float AS current_cost,
    ps.sale_price::float AS current_sale_price,
    g.name AS category,
    COALESCE(s30.sales_last_month, 0) AS sales_last_month,
    COALESCE(st.current_stock, 0) AS current_stock,
    GREATEST(0, EXTRACT(DAY FROM NOW() - p.creation_time))::integer AS product_age_days,
    CASE
        WHEN sa.last_sale_at IS NULL THEN GREATEST(0, EXTRACT(DAY FROM NOW() - p.creation_time))::integer
        ELSE GREATEST(0, EXTRACT(DAY FROM NOW() - sa.last_sale_at))::integer
    END AS days_since_last_sale,
    COALESCE(sa.total_sales_all_time, 0) AS total_sales_all_time
FROM products p
INNER JOIN genders g ON g.id = p.gender_id
INNER JOIN primary_size ps ON ps.product_id = p.id
LEFT JOIN sales_30d s30 ON s30.product_id = p.id
LEFT JOIN sales_all sa ON sa.product_id = p.id
LEFT JOIN stock_totals st ON st.product_id = p.id
WHERE p.is_deleted = false
  AND ps.purchase_price > 0
  AND ps.sale_price > 0
ORDER BY p.id
"""


def main() -> None:
    from app.services.db import read_sql

    ensure_directories()

    logger.info("Exportando series de demanda diaria...")
    demand_df = read_sql(DEMAND_QUERY)
    demand_df.to_csv(DEMAND_DAILY_FILE, index=False)
    logger.info(
        "demand_daily.csv → %d filas, %d productos",
        len(demand_df),
        demand_df["product_id"].nunique() if not demand_df.empty else 0,
    )

    logger.info("Exportando dataset de precios...")
    price_df = read_sql(PRICE_QUERY)
    price_df.to_csv(PRICE_TRAINING_FILE, index=False)
    logger.info(
        "price_training.csv → %d productos",
        len(price_df),
    )


if __name__ == "__main__":
    main()
