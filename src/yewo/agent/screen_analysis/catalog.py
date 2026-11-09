"""Canonical declarations of screenshot-visible UI elements."""

from yewo.agent.screen_analysis.contracts import (
    ScreenElementAffordance,
    ScreenElementCatalog,
    ScreenElementKind as ScreenElementKind,
    ScreenSemanticRole,
    ScreenSemanticRoleDefinition,
    ScreenVisualKind,
    ScreenVisualKindDefinition,
    ScreenWorkflowRole as ScreenWorkflowRole,
)


HEADING = ScreenSemanticRoleDefinition(
    name=ScreenSemanticRole.HEADING,
    allowed_visual_kinds=frozenset({ScreenVisualKind.TEXT}),
    interaction_class=ScreenElementAffordance.NON_INTERACTIVE,
    description="Text that names a screen or divides a visible section.",
    visible_requirements=(
        "The text is visually presented as a screen title or as the boundary of a section or group.",
        "No one-to-one association with an input or other described element is visibly established.",
    ),
)


LABEL = ScreenSemanticRoleDefinition(
    name=ScreenSemanticRole.LABEL,
    allowed_visual_kinds=frozenset({ScreenVisualKind.TEXT}),
    interaction_class=ScreenElementAffordance.NON_INTERACTIVE,
    description="Text that visibly names or describes one other element.",
    visible_requirements=(
        "Alignment, spacing, or a shared container visibly establishes a one-to-one association with another element.",
        "Proximity alone is not sufficient.",
        "The associated element is not established as part of a more specific combined kind.",
    ),
)


TEXT = ScreenSemanticRoleDefinition(
    name=ScreenSemanticRole.TEXT,
    allowed_visual_kinds=frozenset({ScreenVisualKind.TEXT}),
    interaction_class=ScreenElementAffordance.NON_INTERACTIVE,
    description="Visible application text that is not a heading, label, link, or control value.",
    visible_requirements=("Readable application text is visible.",),
)


TEXT_FIELD = ScreenSemanticRoleDefinition(
    name=ScreenSemanticRole.TEXT_FIELD,
    allowed_visual_kinds=frozenset({ScreenVisualKind.INPUT_SURFACE}),
    interaction_class=ScreenElementAffordance.INTERACTIVE,
    description="An editable text or numeric input, including its visibly associated label when present.",
    visible_requirements=(
        "A distinct text-entry surface is visible.",
        "Evidence can be conventional editable styling, an insertion cursor, a placeholder, or one underline that the field uniquely owns.",
        "Text, a border, an underline, or a text label alone is not sufficient without a unique association between the label and editable surface.",
        "One visible input surface can support only one semantic input element.",
        "When several text rows precede one input surface, use grouping, alignment, and form layout to identify one primary label. Distance alone is insufficient.",
        "Classify independently meaningful helper text, validation messages, headings, and unpaired text as separate elements.",
        "Use a more specific input kind when selection-specific or picker-specific evidence is present.",
    ),
)


CHECKBOX = ScreenSemanticRoleDefinition(
    name=ScreenSemanticRole.CHECKBOX,
    allowed_visual_kinds=frozenset({ScreenVisualKind.CONTROL_SURFACE}),
    interaction_class=ScreenElementAffordance.INTERACTIVE,
    description="An independently selectable binary control shown as a box or check mark.",
    visible_requirements=("A checkbox-shaped affordance or check indicator is visible.",),
)


RADIO_BUTTON = ScreenSemanticRoleDefinition(
    name=ScreenSemanticRole.RADIO_BUTTON,
    allowed_visual_kinds=frozenset({ScreenVisualKind.CONTROL_SURFACE}),
    interaction_class=ScreenElementAffordance.INTERACTIVE,
    description="A single-choice control shown as one member of a mutually exclusive group.",
    visible_requirements=(
        "A radio-circle affordance is visible.",
        "At least one related choice or clear group context is visible.",
    ),
)


SWITCH = ScreenSemanticRoleDefinition(
    name=ScreenSemanticRole.SWITCH,
    allowed_visual_kinds=frozenset({ScreenVisualKind.CONTROL_SURFACE}),
    interaction_class=ScreenElementAffordance.INTERACTIVE,
    description="A binary on or off control shown with a track and movable thumb.",
    visible_requirements=("A switch track and thumb are visible.",),
)


SLIDER = ScreenSemanticRoleDefinition(
    name=ScreenSemanticRole.SLIDER,
    allowed_visual_kinds=frozenset({ScreenVisualKind.CONTROL_SURFACE}),
    interaction_class=ScreenElementAffordance.INTERACTIVE,
    description="A control for selecting a value or range along a track.",
    visible_requirements=("A slider track and at least one thumb or value marker are visible.",),
)


DROPDOWN = ScreenSemanticRoleDefinition(
    name=ScreenSemanticRole.DROPDOWN,
    allowed_visual_kinds=frozenset(
        {ScreenVisualKind.INPUT_SURFACE, ScreenVisualKind.TEXT_AND_ICON}
    ),
    interaction_class=ScreenElementAffordance.INTERACTIVE,
    description="A closed or open control for selecting one value from a set of choices.",
    visible_requirements=(
        "A current value or placeholder appears in a selection control with an expansion indicator, attached choice list, or other selection-specific styling.",
        "A field boundary or value alone is not sufficient.",
    ),
)


DATE_PICKER = ScreenSemanticRoleDefinition(
    name=ScreenSemanticRole.DATE_PICKER,
    allowed_visual_kinds=frozenset(
        {ScreenVisualKind.INPUT_SURFACE, ScreenVisualKind.TEXT_AND_ICON}
    ),
    interaction_class=ScreenElementAffordance.INTERACTIVE,
    description="A date input or open date-selection control.",
    visible_requirements=(
        "A date-specific affordance, calendar grid, date selector, or visibly associated date label and input affordance is present.",
        "A formatted date value alone is not sufficient.",
        "Use text_field when no date-specific visual evidence is present.",
    ),
)


TIME_PICKER = ScreenSemanticRoleDefinition(
    name=ScreenSemanticRole.TIME_PICKER,
    allowed_visual_kinds=frozenset(
        {ScreenVisualKind.INPUT_SURFACE, ScreenVisualKind.TEXT_AND_ICON}
    ),
    interaction_class=ScreenElementAffordance.INTERACTIVE,
    description="A time input or open time-selection control.",
    visible_requirements=(
        "A time-specific affordance, clock face, time selector, or visibly associated time label and input affordance is present.",
        "A formatted time value alone is not sufficient.",
        "Use text_field when no time-specific visual evidence is present.",
    ),
)


FILE_PICKER = ScreenSemanticRoleDefinition(
    name=ScreenSemanticRole.FILE_PICKER,
    allowed_visual_kinds=frozenset(
        {ScreenVisualKind.INPUT_SURFACE, ScreenVisualKind.TEXT_AND_ICON}
    ),
    interaction_class=ScreenElementAffordance.INTERACTIVE,
    description="A control that visibly offers file selection or upload.",
    visible_requirements=("A file, attachment, browse, choose, or upload affordance is visible.",),
)


BUTTON = ScreenSemanticRoleDefinition(
    name=ScreenSemanticRole.BUTTON,
    allowed_visual_kinds=frozenset(
        {
            ScreenVisualKind.TEXT,
            ScreenVisualKind.ICON,
            ScreenVisualKind.TEXT_AND_ICON,
            ScreenVisualKind.CONTROL_SURFACE,
        }
    ),
    interaction_class=ScreenElementAffordance.INTERACTIVE,
    description="A visible action control containing text, an icon, or both.",
    visible_requirements=(
        "A button boundary, action styling, or conventional icon-button affordance is visible.",
    ),
)


LINK = ScreenSemanticRoleDefinition(
    name=ScreenSemanticRole.LINK,
    allowed_visual_kinds=frozenset({ScreenVisualKind.TEXT, ScreenVisualKind.TEXT_AND_ICON}),
    interaction_class=ScreenElementAffordance.INTERACTIVE,
    description="Text or content visibly styled as navigation to another destination.",
    visible_requirements=("Link styling or a conventional navigational affordance is visible.",),
)


MENU_ITEM = ScreenSemanticRoleDefinition(
    name=ScreenSemanticRole.MENU_ITEM,
    allowed_visual_kinds=frozenset(
        {ScreenVisualKind.TEXT, ScreenVisualKind.ICON, ScreenVisualKind.TEXT_AND_ICON}
    ),
    interaction_class=ScreenElementAffordance.INTERACTIVE,
    description="One visible action or destination inside an open application menu.",
    visible_requirements=("The item is visibly contained in an open menu.",),
)


OPTION = ScreenSemanticRoleDefinition(
    name=ScreenSemanticRole.OPTION,
    allowed_visual_kinds=frozenset(
        {ScreenVisualKind.TEXT, ScreenVisualKind.TEXT_AND_ICON, ScreenVisualKind.CONTROL_SURFACE}
    ),
    interaction_class=ScreenElementAffordance.INTERACTIVE,
    description="One visible choice inside an open selector.",
    visible_requirements=("The choice is visibly contained in an open selection control.",),
)


CHIP = ScreenSemanticRoleDefinition(
    name=ScreenSemanticRole.CHIP,
    allowed_visual_kinds=frozenset(
        {ScreenVisualKind.TEXT, ScreenVisualKind.TEXT_AND_ICON, ScreenVisualKind.CONTROL_SURFACE}
    ),
    interaction_class=ScreenElementAffordance.CONTEXT_DEPENDENT,
    description="A compact labelled item used for a value, filter, tag, or choice.",
    visible_requirements=("A compact bounded label with chip-like styling is visible.",),
)


TAB = ScreenSemanticRoleDefinition(
    name=ScreenSemanticRole.TAB,
    allowed_visual_kinds=frozenset(
        {ScreenVisualKind.TEXT, ScreenVisualKind.ICON, ScreenVisualKind.TEXT_AND_ICON}
    ),
    interaction_class=ScreenElementAffordance.INTERACTIVE,
    description="A selectable section in a visible tab group.",
    visible_requirements=(
        "Multiple peer section choices or a clear tab-selection indicator are visible.",
    ),
)


NAVIGATION_ITEM = ScreenSemanticRoleDefinition(
    name=ScreenSemanticRole.NAVIGATION_ITEM,
    allowed_visual_kinds=frozenset(
        {ScreenVisualKind.TEXT, ScreenVisualKind.ICON, ScreenVisualKind.TEXT_AND_ICON}
    ),
    interaction_class=ScreenElementAffordance.INTERACTIVE,
    description="One destination in a visible navigation bar, drawer, rail, or application menu.",
    visible_requirements=(
        "The element is visibly grouped with application navigation destinations.",
    ),
)


IMAGE = ScreenSemanticRoleDefinition(
    name=ScreenSemanticRole.IMAGE,
    allowed_visual_kinds=frozenset({ScreenVisualKind.IMAGE}),
    interaction_class=ScreenElementAffordance.CONTEXT_DEPENDENT,
    description="A visible photograph, illustration, avatar, thumbnail, or banner.",
    visible_requirements=("A meaningful raster or illustrated content region is visible.",),
)


ICON = ScreenSemanticRoleDefinition(
    name=ScreenSemanticRole.ICON,
    allowed_visual_kinds=frozenset({ScreenVisualKind.ICON}),
    interaction_class=ScreenElementAffordance.NON_INTERACTIVE,
    description="A visible symbolic graphic without an established action affordance.",
    visible_requirements=(
        "A symbolic graphic is visible.",
        "Use button with visual kind icon when an action affordance is visibly established.",
    ),
)


LIST_ITEM = ScreenSemanticRoleDefinition(
    name=ScreenSemanticRole.LIST_ITEM,
    allowed_visual_kinds=frozenset(
        {
            ScreenVisualKind.TEXT,
            ScreenVisualKind.TEXT_AND_ICON,
            ScreenVisualKind.IMAGE,
            ScreenVisualKind.CONTAINER,
        }
    ),
    interaction_class=ScreenElementAffordance.CONTEXT_DEPENDENT,
    description="One repeated semantic item in a visible list or collection.",
    visible_requirements=("Repeated peer layout or clear list grouping is visible.",),
)


CARD = ScreenSemanticRoleDefinition(
    name=ScreenSemanticRole.CARD,
    allowed_visual_kinds=frozenset({ScreenVisualKind.CONTAINER}),
    interaction_class=ScreenElementAffordance.CONTEXT_DEPENDENT,
    description="A visibly bounded group of related content presented as one item.",
    visible_requirements=(
        "A card boundary, surface, shadow, or clearly grouped content block is visible.",
    ),
)


VIDEO = ScreenSemanticRoleDefinition(
    name=ScreenSemanticRole.VIDEO,
    allowed_visual_kinds=frozenset({ScreenVisualKind.VIDEO}),
    interaction_class=ScreenElementAffordance.CONTEXT_DEPENDENT,
    description="Visible video content or a video player surface.",
    visible_requirements=(
        "A video frame, playback control, timeline, or conventional video indicator is visible.",
    ),
)


PROGRESS_INDICATOR = ScreenSemanticRoleDefinition(
    name=ScreenSemanticRole.PROGRESS_INDICATOR,
    allowed_visual_kinds=frozenset({ScreenVisualKind.INDICATOR}),
    interaction_class=ScreenElementAffordance.NON_INTERACTIVE,
    description="A determinate or indeterminate indicator of progress or loading.",
    visible_requirements=("A progress bar, ring, spinner, or loading animation frame is visible.",),
)


BADGE = ScreenSemanticRoleDefinition(
    name=ScreenSemanticRole.BADGE,
    allowed_visual_kinds=frozenset(
        {ScreenVisualKind.INDICATOR, ScreenVisualKind.TEXT, ScreenVisualKind.TEXT_AND_ICON}
    ),
    interaction_class=ScreenElementAffordance.NON_INTERACTIVE,
    description="A small visible status, count, or notification marker attached to another element.",
    visible_requirements=(
        "A compact attached marker containing a count, status, or notification cue is visible.",
    ),
)


TOOLTIP = ScreenSemanticRoleDefinition(
    name=ScreenSemanticRole.TOOLTIP,
    allowed_visual_kinds=frozenset({ScreenVisualKind.TEXT, ScreenVisualKind.CONTAINER}),
    interaction_class=ScreenElementAffordance.NON_INTERACTIVE,
    description="A small floating explanation visibly associated with another element.",
    visible_requirements=(
        "A floating explanatory surface and visible association with another element are present.",
    ),
)


DIVIDER = ScreenSemanticRoleDefinition(
    name=ScreenSemanticRole.DIVIDER,
    allowed_visual_kinds=frozenset({ScreenVisualKind.DIVIDER}),
    interaction_class=ScreenElementAffordance.NON_INTERACTIVE,
    description="A visible line or separator that divides content regions.",
    visible_requirements=("A deliberate visual separator is visible.",),
)


CONTENT_REGION = ScreenSemanticRoleDefinition(
    name=ScreenSemanticRole.CONTENT_REGION,
    allowed_visual_kinds=frozenset({ScreenVisualKind.CONTAINER, ScreenVisualKind.IMAGE}),
    interaction_class=ScreenElementAffordance.CONTEXT_DEPENDENT,
    description="A meaningful custom-rendered region such as a map, chart, drawing surface, game board, or canvas.",
    visible_requirements=(
        "A bounded or visually coherent custom content surface is visible.",
        "Use a more specific kind when the region is visibly an image or video.",
    ),
)


UNKNOWN = ScreenSemanticRoleDefinition(
    name=ScreenSemanticRole.UNKNOWN,
    allowed_visual_kinds=frozenset(ScreenVisualKind),
    interaction_class=ScreenElementAffordance.CONTEXT_DEPENDENT,
    description="A distinct visible application element whose semantic kind cannot be established reliably.",
    visible_requirements=(
        "A distinct application element is visible.",
        "No defined semantic kind is sufficiently supported by visible evidence.",
        "This kind does not establish that the element is actionable.",
    ),
)


SCREEN_VISUAL_KINDS = (
    ScreenVisualKindDefinition(
        name=ScreenVisualKind.TEXT,
        description="Readable text is the visible surface used to locate the element.",
        visible_requirements=("Readable glyphs form visible text.",),
    ),
    ScreenVisualKindDefinition(
        name=ScreenVisualKind.ICON,
        description="A symbolic glyph is the visible surface used to locate the element.",
        visible_requirements=("A distinct symbolic glyph is visible.",),
    ),
    ScreenVisualKindDefinition(
        name=ScreenVisualKind.TEXT_AND_ICON,
        description="Visible text and an adjacent symbolic glyph jointly identify the element.",
        visible_requirements=("Visible text and a distinct adjacent glyph identify one element.",),
    ),
    ScreenVisualKindDefinition(
        name=ScreenVisualKind.INPUT_SURFACE,
        description="An editable or selection input surface visibly identifies the element.",
        visible_requirements=(
            "A distinct editable surface is visible.",
            "The surface is uniquely associated with one semantic input element.",
            "A label, border, or underline without a unique editable surface is insufficient.",
        ),
        requires_unique_surface=True,
    ),
    ScreenVisualKindDefinition(
        name=ScreenVisualKind.CONTROL_SURFACE,
        description="A bounded control surface, track, box, or chip visibly identifies the element.",
        visible_requirements=("A distinct control boundary, track, box, or chip is visible.",),
    ),
    ScreenVisualKindDefinition(
        name=ScreenVisualKind.IMAGE,
        description="A photograph, illustration, avatar, or thumbnail visibly identifies the element.",
        visible_requirements=("A meaningful raster or illustrated surface is visible.",),
    ),
    ScreenVisualKindDefinition(
        name=ScreenVisualKind.VIDEO,
        description="A video frame or player surface visibly identifies the element.",
        visible_requirements=("A video frame or conventional player surface is visible.",),
    ),
    ScreenVisualKindDefinition(
        name=ScreenVisualKind.CONTAINER,
        description="A bounded region or grouped surface visibly contains other elements.",
        visible_requirements=("A boundary or coherent grouped surface is visible.",),
    ),
    ScreenVisualKindDefinition(
        name=ScreenVisualKind.INDICATOR,
        description="A visible status, progress, badge, or feedback marker identifies the element.",
        visible_requirements=("A distinct status or feedback marker is visible.",),
    ),
    ScreenVisualKindDefinition(
        name=ScreenVisualKind.DIVIDER,
        description="A visible separator line or gap identifies the element.",
        visible_requirements=("A deliberate separator line or gap is visible.",),
    ),
    ScreenVisualKindDefinition(
        name=ScreenVisualKind.UNKNOWN,
        description="The element is visible, but its locating surface kind is uncertain.",
        visible_requirements=("A distinct surface is visible, but its presentation is uncertain.",),
    ),
)


SCREEN_ELEMENT_CATALOG = ScreenElementCatalog(
    name="yewo-visible-element-kinds",
    scope="Semantic kinds that a vision model can establish from one current application screenshot.",
    overlap_rules=(
        "Use label only when visible grouping shows a one-to-one association with another element. Use heading when text names a visible screen, section, or group.",
        "Use a specific input kind instead of text_field only when kind-specific visual evidence is present.",
        "Use link when visible styling shows navigation. Use button when visible styling shows an action.",
        "Use menu_item for an item in an open contextual or action menu. Use navigation_item for an item in application navigation.",
        "Use option for a value choice in an open selector. Use menu_item for an action or destination in an open menu.",
        "Use chip for a compact value, filter, tag, or selection. Use button for a visibly styled action.",
        "Use list_item when repeated peer layout establishes a collection item. Use card when visible boundaries show a content group without repetition.",
        "Use icon for a symbolic graphic without an established action affordance. Use button with visual kind icon when visible evidence shows an action affordance.",
        "Use unknown only when a distinct visible element has insufficient evidence for every defined semantic kind.",
    ),
    interaction_classes={
        ScreenElementAffordance.INTERACTIVE: "The kind conventionally presents a direct interaction affordance. This does not prove current runtime actionability.",
        ScreenElementAffordance.NON_INTERACTIVE: "The kind does not itself present a direct interaction affordance.",
        ScreenElementAffordance.CONTEXT_DEPENDENT: "The kind can be interactive or non-interactive. The screenshot may not establish which behavior applies.",
    },
    visual_kinds=SCREEN_VISUAL_KINDS,
    semantic_roles=(
        HEADING,
        LABEL,
        TEXT,
        TEXT_FIELD,
        CHECKBOX,
        RADIO_BUTTON,
        SWITCH,
        SLIDER,
        DROPDOWN,
        DATE_PICKER,
        TIME_PICKER,
        FILE_PICKER,
        BUTTON,
        LINK,
        MENU_ITEM,
        OPTION,
        CHIP,
        TAB,
        NAVIGATION_ITEM,
        IMAGE,
        ICON,
        LIST_ITEM,
        CARD,
        VIDEO,
        PROGRESS_INDICATOR,
        BADGE,
        TOOLTIP,
        DIVIDER,
        CONTENT_REGION,
        UNKNOWN,
    ),
)


def screen_element_catalog() -> ScreenElementCatalog:
    """Return the canonical typed element catalog."""
    return SCREEN_ELEMENT_CATALOG
