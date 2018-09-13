with cte as (
select tn, churn_losing_sp, churn_date, noncompetitive_ind 
from com_cdm_stage..f_final_churn_reporting where churn_date >= '2018-06-10' and churn_date < '2018-06-13' and churn_losing_sp in (1,2,3,4,6) and churn_type like '%DSV' and noncompetitive_ind in (0,5)
),
cte_minute as (
select churn_losing_sp, datepart(day, churn_date) as cday, datepart(hour,churn_date) as chour, datepart(minute,churn_date) as cminute, count(1) as count_same_minute 
from cte
group by churn_losing_sp, datepart(day, churn_date), datepart(hour,churn_date), datepart(minute,churn_date)
),
cte_second as (
select churn_losing_sp, datepart(day, churn_date) as cday, datepart(hour,churn_date) as chour, datepart(minute,churn_date) as cminute, datepart(second,churn_date) as csecond, count(1) as count_same_second 
from cte
group by churn_losing_sp, datepart(day, churn_date), datepart(hour,churn_date), datepart(minute,churn_date), datepart(second,churn_date) 
)
select a.tn, 
a.churn_date,
case when a.churn_losing_sp = 1 then 1 else 0 end as is_vzw, 
case when a.churn_losing_sp = 2 then 1 else 0 end as is_att, 
case when a.churn_losing_sp = 3 then 1 else 0 end as is_spr, 
case when a.churn_losing_sp = 4 then 1 else 0 end as is_tmo, 
case when a.churn_losing_sp = 6 then 1 else 0 end as is_met,
(DATEPART(HOUR, a.churn_date) * 3600 + DATEPART(MINUTE, a.churn_date) * 60 + DATEPART(SECOND, a.churn_date)) as sec_from_midnight,
case when datepart(hour, a.churn_date) >= 21 or datepart(hour, a.churn_date) < 9 then 1 else 0 end as night_ind,
case when datepart(minute, a.churn_date) % 5 = 0 and datepart(second, a.churn_date) = 0 then 1 else 0 end as five_min_ind,
m.count_same_minute,
s.count_same_second,
case when a.noncompetitive_ind = 5 then 1 else 0 end as noncomp_ind 
from cte a
inner join cte_minute m
on a.CHURN_LOSING_SP = m.CHURN_LOSING_SP
and datepart(day, a.churn_date) = m.cday
and datepart(hour,a.churn_date) = m.chour
and datepart(minute,a.churn_date) = m.cminute
inner join cte_second s
on a.CHURN_LOSING_SP = s.CHURN_LOSING_SP
and datepart(day, a.churn_date) = s.cday
and datepart(hour,a.churn_date) = s.chour
and datepart(minute,a.churn_date) = s.cminute
and datepart(second,a.churn_date) = s.csecond
order by a.churn_losing_sp, a.churn_date






