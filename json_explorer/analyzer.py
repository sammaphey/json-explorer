from __future__ import annotations
from abc import abstractmethod
from calendar import month
from copy import deepcopy
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from math import pi
from typing import Any, Generic, Optional, TypeVar, Union
from bokeh.models import ColumnDataSource, PanTool, HoverTool
from bokeh.plotting import figure, Figure
from datetime import datetime
from dateutil.parser import isoparse

from json_explorer.constants import HANDLED_TYPES, WEEKDAYS


DataType = TypeVar("DataType")


@dataclass()
class TypeAnalyzer(Generic[DataType]):

    data: list[DataType]
    """
    The list of data to analyze.
    
    This list will be post-processed to strip any `None` values.
    """

    original_data: list[DataType] = field(default_factory=lambda: [])
    """The list of data to analyze with contained `None` values."""

    unexpected: list[Any] = field(default_factory=lambda: [])
    """A list of data which matches the key, but is an incorrect type."""

    def __post_init__(self):
        self.original_data = deepcopy(self.data)
        self.data = [d for d in self.data if d is not None]

    @abstractmethod
    def collate(self) -> TypeAnalyzer:
        """Construct all of the collated data used when generating stats."""

    def stats(self) -> str:
        """A markdown formatted representation of the collated statistics."""
        return (
f"""
### {self.__class__.__name__}

- Number of Entries: {len(self.original_data)}
- Number of `null` Entries: {len(self.original_data) - len(self.data)}
- Number of unexpected Entries: {len(self.unexpected)}
"""
        )
    
    @abstractmethod
    def chart(self, key: str) -> figure:
        """A bokeh figure which represents the data in a visualized way."""
        raise NotImplementedError


class StringAnalyzer(TypeAnalyzer[str]):

    _unique: dict[str, int] = field(default_factory=lambda: {})
    """A mapping of the string to the number of times it was found in the list."""

    def collate(self):
        self._unique = Counter(self.data)
        return self
    
    def stats(self):
        return super().stats() + (
f"""
- Number of Unique Values: {len(self._unique)}
"""
        )
    
    def chart(self, key: str) -> Figure:
        if len(self._unique) == len(self.data):
            raise ValueError("Visual Analysis will provide no value. All values are unique.")
        if len(self._unique) == 1:
            raise ValueError("Visual Analysis will provide no value. Only one value to show.")

        keys = list(self._unique.keys())
        counts = list(self._unique.values())

        source = ColumnDataSource(data=dict(keys=keys, counts=counts))

        x_range = keys
        if len(self._unique) > 20:
            x_range = keys[:20]

        p = figure(
            x_range=x_range,
            title=f"{key} Counts",
            tools=[HoverTool()],
            tooltips="@keys, @counts",
            sizing_mode="stretch_width",
        )
        p.xaxis.major_label_orientation = pi/4

        p.vbar(
            x='keys',
            top='counts',
            source=source,
            width=0.9,
            line_color='white',
        )
        p.add_tools(PanTool(dimensions="width"))

        return p


class DateAnalyzer(TypeAnalyzer[datetime]):

    dates: list[datetime]
    """A list of the translated dates."""

    counts: dict[str, dict[str, int]] = field(default_factory=lambda: {})
    """The counts of each day, month, and year combination which appear in the list of dates."""

    max: datetime
    min: datetime

    def __post_init__(self):
        super().__post_init__()
        self.dates = [isoparse(d["$date"]) for d in self.data]

    def collate(self):
        self.min = min(self.dates)
        self.max = max(self.dates)
        self.counts = {
            "day": defaultdict(int),
            "month": defaultdict(int),
            "year": defaultdict(int),
        }
        for date in self.dates:
            day = WEEKDAYS[date.weekday()]
            self.counts["day"][day] += 1
            self.counts["month"][date.month] += 1
            self.counts["year"][date.year] += 1
        return self
    
    def stats(self):
        return super().stats() + (
f"""
- Max Value: {self.max:%c}
- Min Value: {self.min:%c}
- Date Breakdown: {self.counts}
"""
        )


class BooleanAnalyzer(TypeAnalyzer[bool]):

    true: int
    """The number of values that are true in the list."""

    false: int
    """The number of values that are false in the list."""

    def collate(self):
        self.true = len([b for b in self.data if b])
        self.false = len([b for b in self.data if not b])
        return self
    
    def stats(self):
        return super().stats() + (
f"""
- Number of True Values: {self.true}
- Number of False Values: {self.false}
"""
        )
    
    def chart(self, key: str) -> Figure:
        keys = ["true", "false"]
        source = ColumnDataSource(data=dict(keys=keys, counts=[self.true, self.false]))

        p = figure(
            x_range=keys,
            tools=[HoverTool()],
            tooltips="@keys, @counts",
            title=f"{key} Counts",
            sizing_mode="stretch_width",
        )
        p.xaxis.major_label_orientation = pi/4

        p.vbar(
            x='keys',
            top='counts',
            source=source,
            width=0.9,
            line_color='white',
        )

        return p
    

class NumberAnalyzer(TypeAnalyzer[Union[float, int]]):

    max: Union[float, int]
    min: Union[float, int]
    avg: Union[float, int]
    variance: Union[float, int]
    std_dev: Union[float, int]

    def collate(self):
        self.max = max(self.data)
        self.min = min(self.data)
        self.avg = sum(self.data) / len(self.data)
        self.variance = sum([((x - self.avg) ** 2) for x in self.data]) / len(self.data)
        self.std_dev = self.variance ** 0.5
        return self
    
    def stats(self):
        return super().stats() + (
f"""
- Max Value: {self.max}
- Min Value: {self.min}
- Average: {self.avg:.4f}
- Variance: {self.variance:.4f}
- Standard Deviation: {self.std_dev:.4f}
"""
        )


@dataclass(frozen=True)
class Analyzer:
    """Class that handles analyzing a set of similar JSON documents."""

    data: list[dict]
    """The data to initialize the object with."""

    parent: Optional[Analyzer] = None
    """An optional parent analyzer."""

    sub_analyzers: dict[str, Analyzer] = field(default_factory=lambda: {})
    """Sub analyzers for mappings."""

    collated: dict[str, StringAnalyzer] = field(default_factory=lambda: {})
    """A mapping of keys to their collated data."""

    _field_map: dict[str, type] = field(default_factory=lambda: {})
    """The fields present on the objects mapped to their data type."""

    _value_lookup: dict[type, list[str]] = field(default_factory=lambda: defaultdict(list)) 
    """A reverse lookup of the types and fields which contain that type."""

    def analyze(self) -> Analyzer:
        """Analyze the data."""
        # Construct a list of the fields present in the objects
        for key, value in self.data[0].items():

            if type(value) in HANDLED_TYPES and type(value) != dict:
                self.type_analyzer(key, type(value))

            if isinstance(value, dict):
                if "$date" in value and len(value) == 1:
                    self.type_analyzer(key, dict)

                elif not self.parent:
                    self.sub_analyzers[key] = Analyzer(
                        data=[d[key] for d in self.data],
                        parent=self,
                    ).analyze()
        return self
    
    def type_analyzer(self, path: str, type: type):
        self._field_map[path] = type
        self._value_lookup[type].append(path)
        type_dispatch = {
            str: StringAnalyzer,
            float: NumberAnalyzer,
            int: NumberAnalyzer,
            bool: BooleanAnalyzer,
            dict: DateAnalyzer,
        }

        # Get all of the values
        collated = []
        unexpected = []

        for entry in self.data:
            value = entry.get(path, None)
            if value is not None and not isinstance(value, type):
                unexpected.append(value)
            else:
                collated.append(value)            

        # Collate the data
        self.collated[path] = type_dispatch[type](data=collated, unexpected=unexpected).collate()
