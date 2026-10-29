"""The two axes a scenario sits on, and the cells a run covers.

A scenario carries two independent facts: what it looks for, and what shape
it takes. One field can hold only one of them. A category with no type beside
it leaves the shape unsaid. A generator fills that silence by re-walking the
declared journey, which spends a device and adds no evidence.

So the axes are separate. A category says what to look for. A type says what
shape the test takes. Neither decides a scenario alone, and a cell is one
pair of them.

Yewo enumerates the cells from the coverage a file declares, and the
generator fills one cell per call. The model answers how to test one thing.
The file answers what to test. Priority belongs to a product rather than to a
category, so the file carries that too.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class Category(str, Enum):
    """What a scenario looks for."""

    FUNCTIONAL = "functional"
    INPUT = "input"
    SECURITY = "security"
    ACCESSIBILITY = "accessibility"
    PERFORMANCE = "performance"
    NETWORK = "network"
    LIFECYCLE = "lifecycle"
    PERMISSION = "permission"


class ScenarioType(str, Enum):
    """What shape a scenario takes."""

    HAPPY_PATH = "happy_path"
    NEGATIVE = "negative"
    BOUNDARY = "boundary"
    EMPTY = "empty"
    CONCURRENCY = "concurrency"
    INTERRUPTION = "interruption"
    RECOVERY = "recovery"
    ROLE = "role"
    EXPLORATORY = "exploratory"


# A string written under an enum member documents nothing a program can read.
# Python hands back the class docstring for every member, so a prompt built
# from one would tell a model the same sentence eight times. These are the
# descriptions, and a test refuses a member that has none.
CATEGORY_DESCRIPTIONS: dict[Category, str] = {
    Category.FUNCTIONAL: "whether the feature does what the journey says it does",
    Category.INPUT: "what the app does with the values a user enters",
    Category.SECURITY: "whether the app satisfies its declared security rules",
    Category.ACCESSIBILITY: "whether a control can be read, named and hit",
    Category.PERFORMANCE: "how fast the app answered, and whether it stayed responsive",
    Category.NETWORK: "what the app does when the connection changes or goes away",
    Category.LIFECYCLE: "what survives a background, a rotation, a lock or a relaunch",
    Category.PERMISSION: "what the app does when a permission is granted, denied or revoked",
}

TYPE_DESCRIPTIONS: dict[ScenarioType, str] = {
    ScenarioType.HAPPY_PATH: "the declared flow, with the values that are meant to work",
    ScenarioType.NEGATIVE: (
        "an invalid or malformed probe while pursuing the same declared goal"
    ),
    ScenarioType.BOUNDARY: (
        "a high, low, or repeated probe without assuming an undocumented product limit"
    ),
    ScenarioType.EMPTY: (
        "the same declared goal with a relevant value or item deliberately omitted"
    ),
    ScenarioType.CONCURRENCY: "two actions at once, the same action twice, or out of order",
    ScenarioType.INTERRUPTION: "something arrives part way through, and the flow has to survive it",
    ScenarioType.RECOVERY: "something broke, and the app has to come back from it",
    ScenarioType.ROLE: "a different actor, who must see and reach different things",
    ScenarioType.EXPLORATORY: (
        "the same declared goal through an alternate route or action order observed at runtime"
    ),
}


def describe(axis: Category | ScenarioType) -> str:
    """The sentence that tells a model what one axis value means.

    Raises:
        KeyError: If a member was added and left undescribed.
    """
    if isinstance(axis, Category):
        return CATEGORY_DESCRIPTIONS[axis]
    return TYPE_DESCRIPTIONS[axis]


class Priority(str, Enum):
    """Which cells a run has to finish first.

    A run can die part way: a device drops, a provider refuses, a person
    stops it. The order decides what it got through, so a product says which
    cells matter rather than a table in Yewo.
    """

    P0 = "P0"
    P1 = "P1"
    P2 = "P2"


CATEGORIES_AUDITED_EVERYWHERE = frozenset(
    {Category.SECURITY, Category.ACCESSIBILITY, Category.PERFORMANCE}
)
"""Categories whose claims apply to every screen a run passes through.

A masked field has to stay masked on the screen after it. Replay skips
perception on the screens it walks, so these cannot use it.
"""

TYPES_AUDITED_EVERYWHERE = frozenset(
    {ScenarioType.CONCURRENCY, ScenarioType.EXPLORATORY}
)
"""Types that can break a screen they only pass through.

An out-of-order tap and an unscripted one both act where no assertion
expected them, so every screen has to be read.
"""


def must_audit_every_screen(
    category: Category,
    scenario_type: ScenarioType,
    security_rules: Sequence[str] = (),
) -> bool:
    """Whether every screen this scenario walks has to be audited.

    A security rule is only required once some screen reported it as
    applicable. So a screen nobody audits takes its rules out of the
    required set, and a violation there passes. Security rules are declared
    per journey, so this is true of nearly every scenario and not only of
    the cells below.

    Separate from the actions. A scenario may replay the walk and still
    audit what the walk arrives at. What it may never do is skip both.
    """
    return bool(security_rules) or audits_every_screen(category, scenario_type)


def audits_every_screen(category: Category, scenario_type: ScenarioType) -> bool:
    """Whether this cell has to read every screen, rather than replay past them."""
    return (
        category in CATEGORIES_AUDITED_EVERYWHERE
        or scenario_type in TYPES_AUDITED_EVERYWHERE
    )


@dataclass(frozen=True)
class CoverageCell:
    """One pair of axes, at one priority, as a run covers it.

    `index` separates two scenarios asked of the same pair. A file may
    declare a pair twice. So the index is assigned where the cells are built,
    and never defaults into a collision.
    """

    category: Category
    scenario_type: ScenarioType
    priority: Priority
    index: int = 0

    @property
    def key(self) -> str:
        """The name this cell keeps across runs, such as `security.negative.0`.

        Two runs of the same file have to name the same test the same way.
        Without that, a report can only say that two suites disagreed, never
        that one test regressed.
        """
        return f"{self.category.value}.{self.scenario_type.value}.{self.index}"

    @property
    def test_id(self) -> str:
        """The id a report and a JUnit case carry, derived and not invented."""
        return f"TS_{self.category.value}_{self.scenario_type.value}_{self.index}"


DEFAULT_COVERAGE: tuple[CoverageCell, ...] = (
    CoverageCell(Category.FUNCTIONAL, ScenarioType.HAPPY_PATH, Priority.P0),
)
"""What a file covers where it declares no coverage of its own.

One cell, and the most conservative one. A file that names no coverage has
told Yewo nothing about the app. Anything beyond walking the declared
journey would be Yewo inventing work. An app may have no input to refuse, no
security rule to break and no capability to apply. A suite that assumed
otherwise reported on tests that could not run.

Every other cell is declared, which is the contract: one model call per cell
is only affordable where a file says which cells exist.
"""

SUPPORTED_CATEGORIES = frozenset(Category) - {
    Category.PERFORMANCE,
    Category.LIFECYCLE,
}
"""The categories a run can actually produce a scenario for.

`performance` needs a measured number and nothing Yewo records measures one.
`lifecycle` needs the app backgrounded, rotated or locked, and no capability
applies any of those. A cell in either is refused before a model is asked,
rather than answered with a scenario that cannot run.
"""


def index_coverage_cells(
    pairs: list[tuple[Category, ScenarioType, Priority]]
) -> list[CoverageCell]:
    """Cells for declared pairs, numbering a pair that appears more than once.

    A file may ask for the same pair twice, deliberately or by mistake. Two
    cells sharing a key would share a test id. A report could not tell them
    apart, and a later run could not match either one.
    """
    seen: dict[tuple[Category, ScenarioType], int] = {}
    cells = []
    for category, scenario_type, priority in pairs:
        pair = (category, scenario_type)
        index = seen.get(pair, 0)
        seen[pair] = index + 1
        cells.append(CoverageCell(category, scenario_type, priority, index))
    return cells


def sort_cells_by_priority(cells: list[CoverageCell]) -> list[CoverageCell]:
    """The cells in the order a run has to attempt them.

    Priority first, then the order the file declared. A file that lists its
    cells in a deliberate order keeps it inside one priority.
    """
    rank = {priority: index for index, priority in enumerate(Priority)}
    return sorted(cells, key=lambda cell: rank[cell.priority])
