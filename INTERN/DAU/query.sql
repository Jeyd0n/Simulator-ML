SELECT toDate(timestamp) as day, count(DISTINCT user_id) as dau
FROM default.churn_submits 
GROUP BY day