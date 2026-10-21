---
type: Metric
title: Weekly Active Users
description: Count of distinct users with at least one order in a 7-day window.
tags: [engagement, kpi]
timestamp: 2026-05-30T00:00:00Z
---

# Definition

```sql
SELECT COUNT(DISTINCT customer_id) AS wau
FROM `acme.sales.orders`
WHERE placed_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
```

Depends on the [orders table](/tables/orders.md).
