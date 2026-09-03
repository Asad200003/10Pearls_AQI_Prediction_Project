
from feast import Entity
from feast.value_type import ValueType

karachi_location = Entity(
    name="location_id",
    join_keys=["location_id"],
    value_type=ValueType.STRING,   
                                   
    description="Fixed location identifier for the Karachi AQI forecasting site "
                 "(24.8607 N, 67.0011 E). Constant single value: 'karachi'.",
)
