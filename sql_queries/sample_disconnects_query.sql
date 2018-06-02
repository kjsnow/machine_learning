
select 
case when churn_losing_sp = 1 then 1 else 0 end as vzw_ind,
case when churn_losing_sp = 2 then 1 else 0 end as att_ind,
case when churn_losing_sp = 3 then 1 else 0 end as spr_ind,
case when churn_losing_sp = 4 then 1 else 0 end as tmo_ind,
case when churn_losing_sp = 6 then 1 else 0 end as met_ind,
churn_date,
-- churn date converted to seconds after midnight
(DATEPART(HOUR, churn_date) * 3600 + DATEPART(MINUTE, churn_date) * 60 + DATEPART(SECOND, churn_date)) as sec_from_midnight,
-- day/night ind?
case when datepart(hour, churn_date) >= 21 or datepart(hour, churn_date) < 9 then 1 else 0 end as night_ind,
-- five min ind
case when datepart(minute, churn_date) % 5 = 0 and datepart(second, churn_date) = 0 then 1 else 0 end as five_min_ind,
case when noncompetitive_ind = 5 then 1 else 0 end as noncomp_ind
from com_cdm_stage..f_final_churn_reporting
where churn_date >= '2018-05-01' and churn_date < '2018-05-08'
and churn_losing_sp in (1,2,3,4,6) and churn_type like '%DSV'
and noncompetitive_ind in (0,5)

