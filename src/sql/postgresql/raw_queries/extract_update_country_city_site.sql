SELECT 
    orig_id, 
    TRIM(SUBSTRING(orig_id FROM '^.{4}\s(.*?):')) AS extracted_city
FROM evinsights."ChargingStation"
WHERE dataset_id = 7;



UPDATE evinsights."ChargingStation"
SET city = TRIM(SUBSTRING(orig_id FROM '^.{4}\s(.*?):'))
WHERE dataset_id = 7;

UPDATE evinsights."ChargingStation"
SET site_name = TRIM(SUBSTRING(orig_id FROM ':(.*)'))
WHERE dataset_id = 7;


UPDATE evinsights."ChargingStation"
SET country = 'ES'
WHERE dataset_id = 7;


UPDATE evinsights."ChargingStation"
SET country = 'FR'
WHERE dataset_id = 10;


UPDATE evinsights."ChargingStation"
SET country = 'NO'
WHERE dataset_id = 17;