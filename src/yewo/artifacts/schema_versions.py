"""Schema versions for the Yewo artifacts that outlive one run."""

from typing import Literal


# Yewo keeps one version per independently stored artifact. A single shared
# number would reject a readable memory snapshot whenever the procedure format
# changed, and rebuilding memory costs a complete discovery run. The package
# version cannot replace these numbers either, because it also changes for
# releases that leave both formats untouched.
#
# Records that one run writes and the same run reads carry no version. The
# writer and the reader are the same installed build, so they cannot disagree.

AppMemorySchemaVersion = Literal[1]
APP_MEMORY_SCHEMA_VERSION: AppMemorySchemaVersion = 1

CompiledProcedureSchemaVersion = Literal[1]
COMPILED_PROCEDURE_SCHEMA_VERSION: CompiledProcedureSchemaVersion = 1

CanonicalRunRecordSchemaVersion = Literal[1]
CANONICAL_RUN_RECORD_SCHEMA_VERSION: CanonicalRunRecordSchemaVersion = 1
